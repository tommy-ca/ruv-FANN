//! Windows-specific onboarding implementations

use super::*;
use async_trait::async_trait;

/// Windows-specific installer
pub struct WindowsInstaller;

#[async_trait]
impl Installer for WindowsInstaller {
    async fn install(&self, options: InstallOptions) -> Result<PathBuf> {
        // Would implement actual Windows installation logic here
        // - Download MSI or exe installer
        // - Handle UAC elevation if needed
        // - Update PATH environment variable
        todo!("Windows installer implementation")
    }
    
    async fn requires_elevation(&self, target_dir: &Path) -> Result<bool> {
        // Check if target requires admin rights
        match target_dir.to_str() {
            Some(path) if path.contains("Program Files") => Ok(true),
            Some(path) if path.contains("Windows") => Ok(true),
            _ => Ok(false),
        }
    }
    
    async fn verify_installation(&self, path: &Path) -> Result<bool> {
        // On Windows, check if file exists and has .exe extension
        if !path.exists() {
            return Ok(false);
        }
        
        Ok(path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("exe"))
            .unwrap_or(false))
    }
    
    async fn rollback(&self, install_path: &Path) -> Result<()> {
        if install_path.exists() {
            std::fs::remove_file(install_path)
                .map_err(|e| OnboardingError::InstallationFailed(format!("Rollback failed: {}", e)))?;
        }
        Ok(())
    }
}