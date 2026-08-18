import { cp, mkdir, readdir, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const editorRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(editorRoot, 'dist');
const destination = path.resolve(editorRoot, '..', 'frontend', 'public', 'splat-editor');

await mkdir(destination, { recursive: true });
for (const entry of await readdir(destination, { withFileTypes: true })) {
    if (entry.name === 'open.html') continue;
    await rm(path.join(destination, entry.name), { recursive: entry.isDirectory(), force: true });
}
await cp(source, destination, {
    recursive: true,
    filter: pathname => !pathname.endsWith('.map')
});

console.log(`Deployed SuperSplat to ${destination}`);
