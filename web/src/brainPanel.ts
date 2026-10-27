import * as THREE from 'three'
import type { BrainView } from './brainView'

// The fly's brain in a small window pinned to the bottom-left corner. It has its
// own renderer, so it stays visible wherever the main camera goes.

export class BrainPanel {
  private host: HTMLElement
  private renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20)
  private turntable = new THREE.Group()
  private view: BrainView | null = null

  constructor(host: HTMLElement) {
    this.host = host
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    host.appendChild(this.renderer.domElement)
    this.camera.position.set(0, 0.3, 4.4)
    this.camera.lookAt(0, 0, 0)
    this.scene.add(this.turntable)
    new ResizeObserver(() => this.resize()).observe(host)
    this.resize()
  }

  setBrain(view: BrainView) {
    this.view = view
    this.turntable.add(view.points)
  }

  update(dt: number) {
    this.view?.update(dt)
    this.turntable.rotation.y += dt * 0.25
    this.renderer.render(this.scene, this.camera)
  }

  private resize() {
    const width = this.host.clientWidth
    const height = this.host.clientHeight
    if (!width || !height) return
    this.renderer.setSize(width, height)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }
}
