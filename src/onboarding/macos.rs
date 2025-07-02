//! macOS-specific onboarding implementations

use super::*;
use async_trait::async_trait;

/// macOS-specific installer
pub struct MacOSInstaller;

#[async_trait]
impl Installer for MacOSInstaller {
    async fn install(&self, options: InstallOptions) -> Result<PathBuf> {
        // Would implement actual macOS installation logic here
        // - Download .dmg or .pkg file
        // - Handle app bundle installation
        // - Create command-line symlinks
        // - Handle code signing and notarization
        todo!("macOS installer implementation")
    }
    
    async fn requires_elevation(&self, target_dir: &Path) -> Result<bool> {
        // Check if target requires admin rights
        match target_dir.to_str() {
            Some(path) if path.starts_with("/Applications") => Ok(true),
            Some(path) if path.starts_with("/usr/local") => Ok(true),
            _ => Ok(false),
        }
    }
    
    async fn verify_installation(&self, path: &Path) -> Result<bool> {
        use std::os::unix::fs::PermissionsExt;
        
        if !path.exists() {
            return Ok(false);
        }
        
        // Check if it's an app bundle
        if path.extension().and_then(|s| s.to_str()) == Some("app") {
            // Check for executable inside app bundle
            let exe_path = path.join("Contents").join("MacOS");
            return Ok(exe_path.exists());
        }
        
        // Check if executable
        let metadata = std::fs::metadata(path)
            .map_err(|e| OnboardingError::InstallationFailed(format!("Cannot read file metadata: {}", e)))?;
        
        Ok(metadata.permissions().mode() & 0o111 != 0)
    }
    
    async fn rollback(&self, install_path: &Path) -> Result<()> {
        if install_path.exists() {
            if install_path.is_dir() {
                std::fs::remove_dir_all(install_path)
                    .map_err(|e| OnboardingError::InstallationFailed(format!("Rollback failed: {}", e)))?;
            } else {
                std::fs::remove_file(install_path)
                    .map_err(|e| OnboardingError::InstallationFailed(format!("Rollback failed: {}", e)))?;
            }
        }
        Ok(())
    }
}