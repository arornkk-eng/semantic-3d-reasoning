var version = "2.32.3";

const cacheName = `superSplat-v${version}-scene-understanding-1`;
const cacheUrls = [
    './',
    './index.css',
    './index.html',
    './index.js',
    './index.js.map',
    './manifest.json',
    './static/icons/logo-192.png',
    './static/icons/logo-512.png',
    './static/images/screenshot-narrow.jpg',
    './static/images/screenshot-wide.jpg',
    './static/lib/webp/webp.mjs',
    './static/lib/webp/webp.wasm',
    './static/locales/de.json',
    './static/locales/en.json',
    './static/locales/fr.json',
    './static/locales/ja.json',
    './static/locales/ko.json',
    './static/locales/zh-CN.json'
];
self.addEventListener('install', (event) => {
    console.log(`installing v${version}`);
    // create cache for current version
    event.waitUntil(caches.open(cacheName)
        .then((cache) => {
        return cache.addAll(cacheUrls);
    })
        .then(() => self.skipWaiting()));
});
self.addEventListener('activate', (event) => {
    console.log(`activating v${version}`);
    // delete the old caches once this one is activated
    event.waitUntil(caches.keys()
        .then(names => Promise.all(names.filter(name => name !== cacheName).map(name => caches.delete(name))))
        .then(() => self.clients.claim()));
});
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    const liveAsset = event.request.mode === 'navigate' ||
        url.pathname.endsWith('/index.js') || url.pathname.endsWith('/index.css');
    if (liveAsset) {
        event.respondWith(fetch(event.request)
            .then((response) => {
            const copy = response.clone();
            caches.open(cacheName).then(cache => cache.put(event.request, copy));
            return response;
        })
            .catch(() => caches.match(event.request)));
        return;
    }
    event.respondWith(caches.match(event.request)
        .then(response => response ?? fetch(event.request)));
});
