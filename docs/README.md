The process for deploying the docs is currently manual.

They are published to GitHub pages by pushing the generated HTML files to a branch called `gh-pages`,
which gets picked up automatically by GitHub.
The files should live at the root directory of the repository.

The commands to run the deploy are all bundled in the `deploy` Make target, so it's 

```bash
cd docs/
git checkout gh-pages 
git merge master  # See note.
make deploy
```

> [!NOTE]
> It is important to make sure gh-pages is up-to-date so we have the latest docs from mater.
> Otherwise we could be deploying out-of-date docs.

> [!TODO]
> [ ] When deploying remove other files before copying over the HTML files.
> [ ] CI to auto-gen & deploy the update docs on commits to `master`.
