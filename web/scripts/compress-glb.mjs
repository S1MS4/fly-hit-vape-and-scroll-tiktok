// Compress a GLB for the web with meshopt, keeping every UV set, including
// the lightmap UVs (TEXCOORD_1) at enough precision for 4096px lightmaps.
// Usage (from web/): node scripts/compress-glb.mjs in.glb out.glb
import { stat } from 'node:fs/promises'
import { NodeIO } from '@gltf-transform/core'
import { ALL_EXTENSIONS } from '@gltf-transform/extensions'
import { meshopt } from '@gltf-transform/functions'
import { MeshoptDecoder, MeshoptEncoder } from 'meshoptimizer'

const [input, output] = process.argv.slice(2)
if (!input || !output) {
  console.error('usage: node scripts/compress-glb.mjs in.glb out.glb')
  process.exit(1)
}

await Promise.all([MeshoptEncoder.ready, MeshoptDecoder.ready])
const io = new NodeIO()
  .registerExtensions(ALL_EXTENSIONS)
  .registerDependencies({ 'meshopt.encoder': MeshoptEncoder, 'meshopt.decoder': MeshoptDecoder })
const inputBytes = (await stat(input)).size // measured first: output may overwrite input
const document = await io.read(input)
await document.transform(meshopt({ encoder: MeshoptEncoder, level: 'medium', quantizePosition: 16, quantizeTexcoord: 14 }))
await io.write(output, document)
const mb = (bytes) => (bytes / 1e6).toFixed(1)
console.log(`wrote ${output}: ${mb(inputBytes)} MB -> ${mb((await stat(output)).size)} MB`)
