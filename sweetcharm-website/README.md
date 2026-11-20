# Sweetcharm inspiration site

A responsive, static collection gallery inspired by the public Pinterest profile screenshots. It includes searchable and filterable collections, collection detail dialogs, and links back to the profile. It does not take orders or claim to be an official Pinterest page.

## Run locally in VS Code

1. Open this `sweetcharm-website` folder in VS Code.
2. Choose **Terminal → New Terminal**.
3. Run:

   ```powershell
   py -m http.server 8000
   ```

4. Open `http://localhost:8000` in your browser.

Stop the local server with **Ctrl+C** in the terminal.

## Deploy to Vercel

This is a static website with no package installation or build step.

1. Push the project folder to a GitHub repository.
2. In Vercel, choose **Add New → Project** and import that repository.
3. If importing the existing `CodeAlpha_ProjectName` repository, set **Root Directory** to `sweetcharm-website`.
4. Select **Other** as the framework preset, leave the build command blank, and set the output directory to `.`.
5. Select **Deploy**.

The Pinterest destination in this demo uses the profile handle visible in the screenshots. Update the `pinterest.com/simsimtae979/` links in `index.html` if the profile URL changes.
