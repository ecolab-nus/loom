---
id: introduction
title: Loom Documentation
slug: /
hide_title: true
---

# Loom Documentation

Use these guides for detailed information:

- [Architecture](architecture.md): repository layout, subprojects, and the
  compilation pipeline.
- [Docker development](docker.md): image contents, image construction,
  workspace builds, validation, and Tenstorrent hardware access.
- [Native development](development.md): host prerequisites, installation
  options, Python environment management, and build scripts.
- [Usage](usage.md): running kernels, configuration files, writing new
  kernels, and generated outputs.

## Documentation Website

The documentation is configured as a
[Docusaurus](https://docusaurus.io/) docs-only site. Node.js 20 or newer is
required.

Install the site dependencies and start the local development server:

```bash
cd website
npm install
npm start
```

Create a production build:

```bash
cd website
npm run build
npm run serve
```

The generated static site is written to `website/build/`.

The site defaults to `/` and supports deployment-specific URLs through
environment variables. For example, a GitHub Pages project site can be built
with:

```bash
cd website
DOCUSAURUS_URL=https://ecolab-nus.github.io \
DOCUSAURUS_BASE_URL=/loom/ \
npm run build
```
