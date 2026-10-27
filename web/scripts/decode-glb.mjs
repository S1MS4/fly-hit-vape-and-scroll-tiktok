// Decode a compressed glTF (meshopt + quantization) into a plain GLB that
// Blender can import. Usage (from web/): node scripts/decode-glb.mjs in.glb out.glb
import { NodeIO } from '@gltf-transform/core'
import { ALL_EXTENSIONS } from '@gltf-transform/extensions'
import { dequantize } from '@gltf-transform/functions'
import { MeshoptDecoder } from 'meshoptimizer'

const [input, output] = process.argv.slice(2)
if (!input || !output) {
  console.error('usage: node scripts/decode-glb.mjs in.glb out.glb')
  process.exit(1)
}

await MeshoptDecoder.ready
const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({ 'meshopt.decoder': MeshoptDecoder })
const document = await io.read(input)
await document.transform(dequantize())
for (const extension of document.getRoot().listExtensionsUsed()) {
  if (['EXT_meshopt_compression', 'KHR_mesh_quantization'].includes(extension.extensionName)) extension.dispose()
}
await io.write(output, document)
console.log(`wrote ${output}`)
