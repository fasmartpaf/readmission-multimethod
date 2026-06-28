# How to push this to GitHub and get a Zenodo DOI

This folder is already an initialised git repository with one commit. You only need
to create an empty GitHub repo and push.

## 1. Create an empty repo on GitHub
Go to https://github.com/new
- Repository name: `readmission-multimethod`
- Visibility: Public (required for a free Zenodo DOI)
- Do NOT add a README, .gitignore, or license (this repo already has them)

## 2. Push from your computer
Open a terminal in this folder and run (replace YOUR-USERNAME):

```bash
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/readmission-multimethod.git
git push -u origin main
```

If prompted for a password, use a GitHub Personal Access Token
(GitHub → Settings → Developer settings → Personal access tokens), not your account password.

## 3. Mint a DOI on Zenodo
1. Sign in at https://zenodo.org with your GitHub account.
2. Go to https://zenodo.org/account/settings/github/ and flip the switch ON next to
   `readmission-multimethod`.
3. Back on GitHub, create a release: Releases → "Draft a new release" → tag `v1.0.0` → Publish.
4. Zenodo automatically archives the release and issues a DOI (badge appears on the Zenodo page).

## 4. Put the DOI in the paper
Paste the DOI into the manuscript's **Code Availability** statement
(replace the red `[AUTHOR TO PROVIDE ...]` placeholder).

That's it — your code is public, citable, and the paper's data and code availability
requirements are satisfied.
