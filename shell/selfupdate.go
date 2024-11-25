package shell

import (
	"archive/zip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/rs/zerolog/log"
)

func (sc *ShellController) update(cmd *shellcmd) (*Response, error) {
	repo := "domino14/macondo"
	// Step 1: Get the latest release info from GitHub API
	releaseInfo, err := getLatestReleaseInfo(repo)
	if err != nil {
		return nil, fmt.Errorf("failed to get latest release info: %v", err)
	}

	// Step 2: Compare with current version
	currentVersion := sc.macondoVersion
	if releaseInfo.TagName == currentVersion {
		return msg("You are already running the latest version."), nil
	}
	// if currentVersion == "" {
	// 	return msg(`This appears to be a development version of macondo. The self-updater only works for official releases.`), nil
	// }

	// Step 3: Find the asset for the current OS and architecture
	asset, err := findAsset(releaseInfo)
	if err != nil {
		return nil, fmt.Errorf("failed to find a suitable asset for update: %v", err)
	}

	if !(len(cmd.args) > 0 && cmd.args[0] == "install") {
		return msg(fmt.Sprintf("Upgrade available!\n\nTo install macondo %s please type in `update install`.",
			asset.GitTag)), nil
	}

	// Step 4: Download the asset
	zipFilePath, err := downloadAsset(asset)
	if err != nil {
		return nil, fmt.Errorf("failed to download the update: %v", err)
	}

	// Step 5: Extract the zip file
	extractedDir, err := extractZip(zipFilePath)
	if err != nil {
		return nil, fmt.Errorf("failed to extract the update: %v", err)
	}

	// Step 6: Perform the update
	if runtime.GOOS == "windows" {
		err = performWindowsUpdate(extractedDir)
	} else {
		err = performUnixUpdate(extractedDir)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to perform the update: %v", err)
	}

	return msg("Update initiated. Please restart the application."), nil
}

type ReleaseInfo struct {
	TagName string `json:"tag_name"`
	Assets  []struct {
		Name               string `json:"name"`
		BrowserDownloadURL string `json:"browser_download_url"`
	} `json:"assets"`
}

func getLatestReleaseInfo(repo string) (*ReleaseInfo, error) {
	client := &http.Client{Timeout: 10 * time.Second}
	url := fmt.Sprintf("https://api.github.com/repos/%s/releases/latest", repo)

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "macondo-updater")

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var releaseInfo ReleaseInfo
	if err := json.NewDecoder(resp.Body).Decode(&releaseInfo); err != nil {
		return nil, err
	}
	return &releaseInfo, nil
}

type Asset struct {
	Name   string
	URL    string
	GitTag string
}

func findAsset(releaseInfo *ReleaseInfo) (*Asset, error) {
	osName := runtime.GOOS

	var assetName string
	var os string
	switch osName {
	case "windows":
		os = "win64"
	case "darwin":
		os = "osx-universal"
	case "linux":
		os = "linux-x86_64"
	default:
		return nil, errors.New("unsupported and unexpected os: " + osName)
	}

	assetName = fmt.Sprintf("macondo-%s-%s.zip", releaseInfo.TagName, os)

	for _, asset := range releaseInfo.Assets {
		if asset.Name == assetName {
			return &Asset{
				Name:   asset.Name,
				URL:    asset.BrowserDownloadURL,
				GitTag: releaseInfo.TagName,
			}, nil
		}
	}

	return nil, fmt.Errorf("no asset found for OS %s", osName)
}

func downloadAsset(asset *Asset) (string, error) {
	// Download the new executable
	client := &http.Client{Timeout: 0} // No timeout for large downloads
	resp, err := client.Get(asset.URL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Get path of current executable
	exePath, err := os.Executable()
	if err != nil {
		return "", err
	}
	dir := filepath.Dir(exePath)

	// Download to a temporary file
	tmpFile, err := os.CreateTemp(dir, "macondo-update-*")
	if err != nil {
		return "", err
	}
	defer tmpFile.Close()

	_, err = io.Copy(tmpFile, resp.Body)
	if err != nil {
		return "", err
	}

	// On Unix systems, set the executable permission
	if runtime.GOOS != "windows" {
		if err := os.Chmod(tmpFile.Name(), 0755); err != nil {
			return "", err
		}
	}

	return tmpFile.Name(), nil
}

func performUnixUpdate(extractedDir string) error {
	exePath, err := os.Executable()
	if err != nil {
		return err
	}
	log.Info().Str("exePath", exePath).Msg("current-executable")

	// Path to the new executable
	newExePath := filepath.Join(extractedDir, filepath.Base(exePath))

	// Replace the current executable by copying
	err = replaceExecutable(newExePath, exePath)
	if err != nil {
		return err
	}

	// Merge the data directory
	dataDirPath := filepath.Join(filepath.Dir(exePath), "data")
	newDataDirPath := filepath.Join(extractedDir, "data")
	err = MergeDirectories(newDataDirPath, dataDirPath)
	if err != nil {
		return err
	}

	return nil
}

func replaceExecutable(src, dest string) error {
	// Copy the new executable over the existing one
	err := copyFile(src, dest)
	if err != nil {
		return err
	}

	// Set the executable permissions on the destination file
	err = os.Chmod(dest, 0755)
	if err != nil {
		return err
	}

	return nil
}

func performWindowsUpdate(extractedDir string) error {
	exePath, err := os.Executable()
	if err != nil {
		return err
	}

	dataDir := filepath.Join(filepath.Dir(exePath), "data")

	// Path to the updater executable
	updaterPath := filepath.Join(filepath.Dir(exePath), "updater.exe")

	// Check if the updater executable exists
	if _, err := os.Stat(updaterPath); os.IsNotExist(err) {
		return fmt.Errorf("updater executable not found at %s", updaterPath)
	}

	// Execute the updater
	cmd := exec.Command(updaterPath, exePath, extractedDir, dataDir)
	err = cmd.Start()
	if err != nil {
		return err
	}
	fmt.Println("Press Enter to exit the application and begin the update process...")
	fmt.Scanln()
	// Exit the application
	os.Exit(0)
	return nil
}

// Extracts the zip file to a temporary directory and returns the path
func extractZip(zipFilePath string) (string, error) {
	// Create a temporary directory to extract to
	tmpDir, err := os.MkdirTemp("", "macondo-extract-*")
	if err != nil {
		return "", err
	}

	r, err := zip.OpenReader(zipFilePath)
	if err != nil {
		return "", err
	}
	defer r.Close()

	// Iterate through each file in the zip archive
	for _, f := range r.File {
		fpath := filepath.Join(tmpDir, f.Name)

		// Check for ZipSlip vulnerability
		if !strings.HasPrefix(fpath, filepath.Clean(tmpDir)+string(os.PathSeparator)) {
			return "", fmt.Errorf("illegal file path: %s", fpath)
		}

		if f.FileInfo().IsDir() {
			// Make Folder
			os.MkdirAll(fpath, os.ModePerm)
			continue
		}

		// Make File
		if err = os.MkdirAll(filepath.Dir(fpath), os.ModePerm); err != nil {
			return "", err
		}

		outFile, err := os.OpenFile(fpath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, f.Mode())
		if err != nil {
			return "", err
		}

		rc, err := f.Open()
		if err != nil {
			return "", err
		}

		_, err = io.Copy(outFile, rc)

		// Close the file without defer to close before next iteration
		outFile.Close()
		rc.Close()

		if err != nil {
			return "", err
		}
	}

	return tmpDir, nil
}

func MergeDirectories(srcDir, destDir string) error {
	entries, err := os.ReadDir(srcDir)
	if err != nil {
		return err
	}

	for _, entry := range entries {
		srcPath := filepath.Join(srcDir, entry.Name())
		destPath := filepath.Join(destDir, entry.Name())

		if entry.IsDir() {
			// Recursively merge directories
			if err := MergeDirectories(srcPath, destPath); err != nil {
				return err
			}
		} else {
			// Copy files
			err := copyFile(srcPath, destPath)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func copyFile(srcFile, destFile string) error {
	input, err := os.Open(srcFile)
	if err != nil {
		return err
	}
	defer input.Close()

	// Ensure the destination directory exists
	destDir := filepath.Dir(destFile)
	if err := os.MkdirAll(destDir, os.ModePerm); err != nil {
		return err
	}

	// Create or truncate the destination file
	output, err := os.OpenFile(destFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
	if err != nil {
		return err
	}
	defer output.Close()

	_, err = io.Copy(output, input)
	if err != nil {
		return err
	}

	return nil
}
