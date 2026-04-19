import esbuild from 'esbuild';
import { rmSync, statSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import copyStaticFiles from 'esbuild-copy-static-files';

const codeExtensions = new Set(['.ts', '.tsx', '.js', '.jsx', '.mts', '.cts']);
const bundledAssetExtensions = new Set([
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.webp',
  '.svg',
  '.mp3',
  '.wav',
  '.ogg',
  '.ttf',
  '.woff',
  '.woff2',
]);
const watch = process.argv.includes('--watch');

const buildOptions = {
  entryPoints: ['src/game.ts'],
  bundle: true,
  external: ['fs', 'path'],
  minify: !watch,
  sourcemap: watch ? 'inline' : false,
  assetNames: '[name]',
  outfile: 'build/game.js',
  publicPath: 'build',
  loader: Object.fromEntries([...bundledAssetExtensions].map((extension) => [extension, 'file'])),
  plugins: [
    copyStaticFiles({
      src: './src',
      dest: './build',
      recursive: true,
      filter: (sourcePath) => {
        if (statSync(sourcePath).isDirectory()) return true;

        const extension = path.extname(sourcePath).toLowerCase();
        return !codeExtensions.has(extension) && !bundledAssetExtensions.has(extension);
      },
    }),
    // copy MediaPipe runtime files from node_modules into build so locateFile can load them locally
    copyStaticFiles({
      src: './node_modules/@mediapipe/hands',
      dest: './build/@mediapipe/hands',
    }),
    copyStaticFiles({
      src: './node_modules/littlejsengine/plugins/box2d.wasm.js',
      dest: './build/box2d.wasm.js',
    }),
    copyStaticFiles({
      src: './node_modules/littlejsengine/plugins/box2d.wasm.wasm',
      dest: './build/box2d.wasm.wasm',
    }),
  ],
};

if (watch) {
  const context = await esbuild.context(buildOptions);
  await context.watch();
} else {
  await esbuild.build(buildOptions);
  rmSync('build.zip', { force: true });
  execFileSync('zip', ['-rq', 'build.zip', 'index.html', 'build']);
}
