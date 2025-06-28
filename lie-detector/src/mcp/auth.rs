//! Authentication and Authorization Module for Veritas MCP Server
//!
//! This module provides comprehensive authentication and authorization capabilities
//! including API key authentication, mTLS support, and role-based access control.

use anyhow::{Context, Result};
use bcrypt::{hash, verify, DEFAULT_COST};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::mcp::{McpError, McpResult};

/// Authentication levels for role-based access control
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthLevel {
    /// Read-only access - view models, results, and metrics
    ReadOnly,
    /// Analyst access - perform inference and analysis
    Analyst,
    /// Trainer access - train and manage models
    Trainer,
    /// Admin access - full system configuration
    Admin,
}

impl AuthLevel {
    /// Check if this auth level permits the given operation
    pub fn permits(&self, required: AuthLevel) -> bool {
        let self_level = self.level_value();
        let required_level = required.level_value();
        self_level >= required_level
    }

    fn level_value(&self) -> u8 {
        match self {
            AuthLevel::ReadOnly => 0,
            AuthLevel::Analyst => 1,
            AuthLevel::Trainer => 2,
            AuthLevel::Admin => 3,
        }
    }
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// JWT secret key for token signing
    pub jwt_secret: String,
    /// Token expiration time in seconds
    pub token_expiry_seconds: u64,
    /// Enable API key authentication
    pub enable_api_keys: bool,
    /// Enable mTLS authentication
    pub enable_mtls: bool,
    /// Path to CA certificate for mTLS
    pub ca_cert_path: Option<String>,
    /// Path to server certificate
    pub server_cert_path: Option<String>,
    /// Path to server private key
    pub server_key_path: Option<String>,
    /// Maximum failed authentication attempts
    pub max_failed_attempts: u32,
    /// Lockout duration in seconds after max attempts
    pub lockout_duration_seconds: u64,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: "default-secret-change-in-production".to_string(),
            token_expiry_seconds: 3600, // 1 hour
            enable_api_keys: true,
            enable_mtls: false,
            ca_cert_path: None,
            server_cert_path: None,
            server_key_path: None,
            max_failed_attempts: 5,
            lockout_duration_seconds: 300, // 5 minutes
        }
    }
}

/// Authentication information for a validated request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthInfo {
    pub user_id: String,
    pub auth_level: AuthLevel,
    pub session_id: String,
    pub issued_at: u64,
    pub expires_at: u64,
    pub client_id: Option<String>,
    pub scopes: Vec<String>,
}

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Claims {
    sub: String, // user_id
    level: AuthLevel,
    session_id: String,
    iat: u64, // issued at
    exp: u64, // expiration
    scopes: Vec<String>,
}

/// API key information
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKey {
    key_id: String,
    key_hash: String,
    user_id: String,
    auth_level: AuthLevel,
    created_at: u64,
    last_used: Option<u64>,
    is_active: bool,
    scopes: Vec<String>,
}

/// Failed authentication attempt tracking
#[derive(Debug, Clone)]
struct FailedAttempt {
    count: u32,
    last_attempt: SystemTime,
    locked_until: Option<SystemTime>,
}

/// Main authentication manager
#[derive(Debug)]
pub struct AuthManager {
    config: AuthConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    api_keys: RwLock<HashMap<String, ApiKey>>,
    failed_attempts: RwLock<HashMap<String, FailedAttempt>>,
    active_sessions: RwLock<HashMap<String, AuthInfo>>,
}

impl AuthManager {
    /// Create a new authentication manager
    pub async fn new(config: AuthConfig) -> Result<Self> {
        info!("Initializing authentication manager");

        let encoding_key = EncodingKey::from_secret(config.jwt_secret.as_bytes());
        let decoding_key = DecodingKey::from_secret(config.jwt_secret.as_bytes());

        let manager = Self {
            config,
            encoding_key,
            decoding_key,
            api_keys: RwLock::new(HashMap::new()),
            failed_attempts: RwLock::new(HashMap::new()),
            active_sessions: RwLock::new(HashMap::new()),
        };

        // Initialize with default admin user if no keys exist
        manager.ensure_admin_access().await?;

        info!("Authentication manager initialized successfully");
        Ok(manager)
    }

    /// Validate an authentication token (JWT or API key)
    pub async fn validate_token(&self, auth_header: &str) -> McpResult<AuthInfo> {
        if let Some(token) = auth_header.strip_prefix("Bearer ") {
            self.validate_jwt_token(token).await
        } else if let Some(api_key) = auth_header.strip_prefix("ApiKey ") {
            self.validate_api_key(api_key).await
        } else {
            Err(McpError::AuthFailed {
                reason: "Invalid authentication header format".to_string(),
            })
        }
    }

    /// Validate a JWT token
    async fn validate_jwt_token(&self, token: &str) -> McpResult<AuthInfo> {
        let validation = Validation::new(Algorithm::HS256);
        
        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)
            .map_err(|e| McpError::AuthFailed {
                reason: format!("Invalid JWT token: {}", e),
            })?;

        let claims = token_data.claims;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        if claims.exp < now {
            return Err(McpError::AuthFailed {
                reason: "Token has expired".to_string(),
            });
        }

        // Check if session is still active
        let sessions = self.active_sessions.read().await;
        if let Some(session) = sessions.get(&claims.session_id) {
            Ok(session.clone())
        } else {
            // Token is valid but session doesn't exist, create new session info
            let auth_info = AuthInfo {
                user_id: claims.sub,
                auth_level: claims.level,
                session_id: claims.session_id,
                issued_at: claims.iat,
                expires_at: claims.exp,
                client_id: None,
                scopes: claims.scopes,
            };
            
            // Store session for future reference
            drop(sessions);
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(auth_info.session_id.clone(), auth_info.clone());
            
            Ok(auth_info)
        }
    }

    /// Validate an API key
    async fn validate_api_key(&self, key: &str) -> McpResult<AuthInfo> {
        let api_keys = self.api_keys.read().await;
        
        // Find the API key by checking all stored keys
        for (_, api_key_info) in api_keys.iter() {
            if verify(key, &api_key_info.key_hash).unwrap_or(false) {
                if !api_key_info.is_active {
                    return Err(McpError::AuthFailed {
                        reason: "API key is disabled".to_string(),
                    });
                }

                // Update last used timestamp
                drop(api_keys);
                let mut api_keys = self.api_keys.write().await;
                if let Some(api_key) = api_keys.get_mut(&api_key_info.key_id) {
                    api_key.last_used = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                    );
                }

                let session_id = Uuid::new_v4().to_string();
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                return Ok(AuthInfo {
                    user_id: api_key_info.user_id.clone(),
                    auth_level: api_key_info.auth_level,
                    session_id,
                    issued_at: now,
                    expires_at: now + self.config.token_expiry_seconds,
                    client_id: Some(api_key_info.key_id.clone()),
                    scopes: api_key_info.scopes.clone(),
                });
            }
        }

        Err(McpError::AuthFailed {
            reason: "Invalid API key".to_string(),
        })
    }

    /// Generate a new JWT token for authenticated user
    pub async fn generate_token(
        &self,
        user_id: &str,
        auth_level: AuthLevel,
        scopes: Vec<String>,
    ) -> McpResult<String> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let session_id = Uuid::new_v4().to_string();
        
        let claims = Claims {
            sub: user_id.to_string(),
            level: auth_level,
            session_id: session_id.clone(),
            iat: now,
            exp: now + self.config.token_expiry_seconds,
            scopes: scopes.clone(),
        };

        let header = Header::new(Algorithm::HS256);
        let token = encode(&header, &claims, &self.encoding_key)
            .map_err(|e| McpError::AuthFailed {
                reason: format!("Failed to generate token: {}", e),
            })?;

        // Store session info
        let auth_info = AuthInfo {
            user_id: user_id.to_string(),
            auth_level,
            session_id: session_id.clone(),
            issued_at: now,
            expires_at: claims.exp,
            client_id: None,
            scopes,
        };

        let mut sessions = self.active_sessions.write().await;
        sessions.insert(session_id, auth_info);

        Ok(token)
    }

    /// Create a new API key
    pub async fn create_api_key(
        &self,
        user_id: &str,
        auth_level: AuthLevel,
        scopes: Vec<String>,
    ) -> McpResult<String> {
        let key = Uuid::new_v4().to_string();
        let key_hash = hash(&key, DEFAULT_COST)
            .map_err(|e| McpError::Internal {
                message: format!("Failed to hash API key: {}", e),
            })?;

        let api_key = ApiKey {
            key_id: Uuid::new_v4().to_string(),
            key_hash,
            user_id: user_id.to_string(),
            auth_level,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_used: None,
            is_active: true,
            scopes,
        };

        let mut api_keys = self.api_keys.write().await;
        api_keys.insert(api_key.key_id.clone(), api_key);

        info!("Created API key for user: {}", user_id);
        Ok(key)
    }

    /// Revoke an API key
    pub async fn revoke_api_key(&self, key_id: &str) -> McpResult<()> {
        let mut api_keys = self.api_keys.write().await;
        
        if let Some(api_key) = api_keys.get_mut(key_id) {
            api_key.is_active = false;
            info!("Revoked API key: {}", key_id);
            Ok(())
        } else {
            Err(McpError::AuthFailed {
                reason: "API key not found".to_string(),
            })
        }
    }

    /// Revoke a session
    pub async fn revoke_session(&self, session_id: &str) -> McpResult<()> {
        let mut sessions = self.active_sessions.write().await;
        
        if sessions.remove(session_id).is_some() {
            info!("Revoked session: {}", session_id);
            Ok(())
        } else {
            Err(McpError::AuthFailed {
                reason: "Session not found".to_string(),
            })
        }
    }

    /// Check if a client is currently locked out
    pub async fn is_locked_out(&self, client_id: &str) -> bool {
        let failed_attempts = self.failed_attempts.read().await;
        
        if let Some(attempt) = failed_attempts.get(client_id) {
            if let Some(locked_until) = attempt.locked_until {
                return SystemTime::now() < locked_until;
            }
        }
        
        false
    }

    /// Record a failed authentication attempt
    pub async fn record_failed_attempt(&self, client_id: &str) {
        let mut failed_attempts = self.failed_attempts.write().await;
        let now = SystemTime::now();
        
        let attempt = failed_attempts.entry(client_id.to_string()).or_insert(FailedAttempt {
            count: 0,
            last_attempt: now,
            locked_until: None,
        });

        attempt.count += 1;
        attempt.last_attempt = now;

        if attempt.count >= self.config.max_failed_attempts {
            attempt.locked_until = Some(now + Duration::from_secs(self.config.lockout_duration_seconds));
            warn!("Client {} locked out due to too many failed attempts", client_id);
        }
    }

    /// Clear failed attempts for a client (on successful authentication)
    pub async fn clear_failed_attempts(&self, client_id: &str) {
        let mut failed_attempts = self.failed_attempts.write().await;
        failed_attempts.remove(client_id);
    }

    /// Cleanup expired sessions and lockouts
    pub async fn cleanup_expired(&self) -> Result<()> {
        let now = SystemTime::now();
        let now_unix = now.duration_since(UNIX_EPOCH).unwrap().as_secs();

        // Clean up expired sessions
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.retain(|_, session| session.expires_at > now_unix);
        }

        // Clean up expired lockouts
        {
            let mut failed_attempts = self.failed_attempts.write().await;
            failed_attempts.retain(|_, attempt| {
                if let Some(locked_until) = attempt.locked_until {
                    now < locked_until
                } else {
                    true
                }
            });
        }

        debug!("Cleaned up expired authentication data");
        Ok(())
    }

    /// Ensure admin access is available by creating a default admin API key
    async fn ensure_admin_access(&self) -> Result<()> {
        let api_keys = self.api_keys.read().await;
        
        // Check if any admin keys exist
        let has_admin = api_keys.values().any(|key| {
            key.auth_level == AuthLevel::Admin && key.is_active
        });

        if !has_admin {
            drop(api_keys);
            
            let admin_key = self.create_api_key(
                "admin",
                AuthLevel::Admin,
                vec!["*".to_string()], // All scopes
            ).await?;

            warn!("Created default admin API key: {}", admin_key);
            warn!("Please change this key in production!");
        }

        Ok(())
    }

    /// Get authentication statistics
    pub async fn get_auth_stats(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        let sessions = self.active_sessions.read().await;
        let api_keys = self.api_keys.read().await;
        let failed_attempts = self.failed_attempts.read().await;

        stats.insert("active_sessions".to_string(), serde_json::json!(sessions.len()));
        stats.insert("total_api_keys".to_string(), serde_json::json!(api_keys.len()));
        stats.insert("active_api_keys".to_string(), serde_json::json!(
            api_keys.values().filter(|k| k.is_active).count()
        ));
        stats.insert("failed_attempts".to_string(), serde_json::json!(failed_attempts.len()));
        stats.insert("locked_clients".to_string(), serde_json::json!(
            failed_attempts.values().filter(|a| a.locked_until.is_some()).count()
        ));

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_manager_creation() {
        let config = AuthConfig::default();
        let auth_manager = AuthManager::new(config).await;
        assert!(auth_manager.is_ok());
    }

    #[test]
    fn test_auth_level_permissions() {
        assert!(AuthLevel::Admin.permits(AuthLevel::ReadOnly));
        assert!(AuthLevel::Admin.permits(AuthLevel::Analyst));
        assert!(AuthLevel::Admin.permits(AuthLevel::Trainer));
        assert!(AuthLevel::Admin.permits(AuthLevel::Admin));
        
        assert!(!AuthLevel::ReadOnly.permits(AuthLevel::Analyst));
        assert!(!AuthLevel::ReadOnly.permits(AuthLevel::Trainer));
        assert!(!AuthLevel::ReadOnly.permits(AuthLevel::Admin));
        
        assert!(AuthLevel::Trainer.permits(AuthLevel::ReadOnly));
        assert!(AuthLevel::Trainer.permits(AuthLevel::Analyst));
        assert!(AuthLevel::Trainer.permits(AuthLevel::Trainer));
        assert!(!AuthLevel::Trainer.permits(AuthLevel::Admin));
    }

    #[tokio::test]
    async fn test_api_key_creation() {
        let config = AuthConfig::default();
        let auth_manager = AuthManager::new(config).await.unwrap();
        
        let key = auth_manager.create_api_key(
            "test_user",
            AuthLevel::Analyst,
            vec!["inference".to_string()],
        ).await;
        
        assert!(key.is_ok());
        assert!(!key.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_token_generation() {
        let config = AuthConfig::default();
        let auth_manager = AuthManager::new(config).await.unwrap();
        
        let token = auth_manager.generate_token(
            "test_user",
            AuthLevel::Analyst,
            vec!["inference".to_string()],
        ).await;
        
        assert!(token.is_ok());
        assert!(!token.unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_lockout_mechanism() {
        let config = AuthConfig {
            max_failed_attempts: 2,
            lockout_duration_seconds: 60,
            ..Default::default()
        };
        let auth_manager = AuthManager::new(config).await.unwrap();
        let client_id = "test_client";
        
        // Initially not locked out
        assert!(!auth_manager.is_locked_out(client_id).await);
        
        // Record failed attempts
        auth_manager.record_failed_attempt(client_id).await;
        assert!(!auth_manager.is_locked_out(client_id).await);
        
        auth_manager.record_failed_attempt(client_id).await;
        assert!(auth_manager.is_locked_out(client_id).await);
        
        // Clear attempts
        auth_manager.clear_failed_attempts(client_id).await;
        assert!(!auth_manager.is_locked_out(client_id).await);
    }
}