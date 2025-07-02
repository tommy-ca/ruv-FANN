//! Linux-specific onboarding implementations

use super::*;
use async_trait::async_trait;

/// Linux-specific installer
pub struct LinuxInstaller;

#[async_trait]
impl Installer for LinuxInstaller {
    async fn install(&self, options: InstallOptions) -> Result<PathBuf> {
        // Would implement actual Linux installation logic here
        // - Download from GitHub releases or package manager
        // - Handle different package formats (.deb, .rpm, .tar.gz)
        // - Set up proper permissions
        todo!("Linux installer implementation")
    }
    
    async fn requires_elevation(&self, target_dir: &Path) -> Result<bool> {
        // Check if target directory requires sudo
        match target_dir.to_str() {
            Some(path) if path.starts_with("/usr") => Ok(true),
            Some(path) if path.starts_with("/opt") => Ok(true),
            _ => Ok(false),
        }
    }
    
    async fn verify_installation(&self, path: &Path) -> Result<bool> {
        use std::os::unix::fs::PermissionsExt;
        
        if !path.exists() {
            return Ok(false);
        }
        
        // Check if executable
        let metadata = std::fs::metadata(path)
            .map_err(|e| OnboardingError::InstallationFailed(format!("Cannot read file metadata: {}", e)))?;
        
        Ok(metadata.permissions().mode() & 0o111 != 0)
    }
    
    async fn rollback(&self, install_path: &Path) -> Result<()> {
        if install_path.exists() {
            std::fs::remove_file(install_path)
                .map_err(|e| OnboardingError::InstallationFailed(format!("Rollback failed: {}", e)))?;
        }
        Ok(())
    }
}