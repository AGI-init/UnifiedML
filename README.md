# Quick tutorial 

These are the documentation files for [UnifiedML](https://github.com/agi-init/UnifiedML), deployed live [here](https://agi-init.github.io/UnifiedML). 

### Step by step instructions for how to generate one's own documentation like this on one's own GitHub

1. Create a Docs branch in your GitHub.
2. In your console, on that branch locally, run:
    - ``mkdir docs``
    - ``cd docs``
    - ``pip install sphynx``
    - ``pip install sphinx_rtd_theme``
    - ``sphinx-quickstart``
3. Accept default (select [n]), then enter your own details.
4. Modify the created ``conf.py`` with:
```python
   extensions = [’sphinx_rtd_theme’]
   ...
   html_theme = ’sphinx_rtd_theme’
```
5. Create your documentation by modifying ``index.rst`` and adding src RST files. For more info about creating documentation with RST files, see [Sphynx documentation](https://www.sphinx-doc.org/en/master/).
6. Preview your work with ``make html`` and then opening the generated HTML files in your local browser.
7. In the root directory (``cd ..``) on this branch locally, create a GitHub workflow:
    - ``mkdir .github/``
    - ``mkdir .github/workflows/``
    - ``vim .github/workflows/Docs.yaml``
    - with the following workflow:
```yaml
name: Docs
on: [push, pull_request, workflow_dispatch]
permissions:
    contents: write
jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v3
      - name: Install dependencies
        run: |
          pip install furo==2023.7.26
          pip install sphinxemoji
          pip install sphinx sphinx_rtd_theme
      - name: Sphinx build
        run: |
          sphinx-build ./docs _build
      - name: Deploy
        uses: peaceiris/actions-gh-pages@v3
        if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/Docs' }}
        with:
          publish_branch: Pages
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: _build/
          force_orphan: true
```
8. Push created directory files to the Docs branch on GitHub (add, commit, push).
9. A "Pages" branch should be automatically created. Add "Pages" branch to GitHub Pages: ``“Settings” —> “Pages” —> Pages``.
10. View your docs live at ``https://<my_username>.github.io/<my_repo_name>``.

## Please consider [donating](https://www.github.com/sponsors/agi-init).