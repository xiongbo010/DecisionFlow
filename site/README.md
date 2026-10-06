# DecisionFlow static site

The site is a dependency-free HTML, CSS, and JavaScript build. Preview it with:

```bash
cd site
python3 -m http.server 8080
```

Then open `http://localhost:8080/`. The directory can be deployed independently because repository documents use absolute GitHub links.

## Files

- `index.html`: page structure and project directory;
- `demo.html`: interactive service-ticket inference demo;
- `demo.css`: responsive demo-specific visual system;
- `demo.js`: exact browser-side inference over the demo decision worlds;
- `styles.css`: light visual system and responsive layouts;
- `script.js`: mobile navigation, ecosystem filters, reveal transitions, and code copying;
- `favicon.svg`: DecisionFlow mark.

No build step, package manager, external font, analytics script, or remote runtime dependency is required.
