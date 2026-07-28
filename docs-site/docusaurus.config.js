const config = {
  title: 'Loom',
  tagline: 'An end-to-end compilation stack for spatial hardware',
  url: process.env.DOCUSAURUS_URL || 'https://ecolab-nus.github.io',
  baseUrl: process.env.DOCUSAURUS_BASE_URL || '/',
  organizationName: 'ecolab-nus',
  projectName: 'loom',
  onBrokenLinks: 'throw',

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: '/',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/ecolab-nus/loom/edit/main/docs/',
        },
        blog: false,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Loom',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/ecolab-nus/loom',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Architecture', to: '/architecture'},
            {label: 'Docker', to: '/docker'},
            {label: 'Usage', to: '/usage'},
          ],
        },
        {
          title: 'Project',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/ecolab-nus/loom',
            },
            {
              label: 'Docker Hub',
              href: 'https://hub.docker.com/r/ftod/loom_dev',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Loom contributors.`,
    },
  },
};

module.exports = config;
