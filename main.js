"use strict";


const PRIME_A = 134053;
const PRIME_B = 200183;
const PRIME_C = 145931;
const PRIME_D = 161387;
// const MASK = (1 << 48) - 1;


function randomize(seed) {
	return (seed * PRIME_A + PRIME_B) ^ 1999998827;
	// let num: u64 = (seed as u64 ^ MULTIPLIER) & MASK;
	// (((num.wrapping_mul(MULTIPLIER).wrapping_add(ADDEND)) & MASK) >> 16) as u32
}

class WhiteNoise {
	constructor(seed) {
		this.seed = randomize(randomize(seed));
	}

	hashPos(x, y) {
		return randomize(this.seed ^ randomize(randomize(x*3) ^ randomize(y*89)));
	}
}


class MapGen {
	constructor(seed, width, height) {
		this.width = width;
		this.height = height;
		// this.noise = new WhiteNoise(seed);
		this.noise = new FastNoiseLite();
		this.noise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2);
		this.noise.SetFractalType(FastNoiseLite.FractalType.FBm);
		this.noise.SetFractalOctaves(8);
		this.noise.SetFrequency(0.005);
	}


	colorAt(x, y) {
		return this.heightAt(x, y) * 256;
	}

	heightAt(x, y) {
		let nx = 2*x/this.width - 1;
		let ny = 2*y/this.height - 1;
		// let d = Math.min(1, (nx*nx + ny*ny) / 1.4142135623730951);
		let d = 1 - (1-nx*nx) * (1-ny * ny);
		let e = this.noise.GetNoise(x, y)// *0.5 + 0.5;
		e = (e + 1 - 2*d)/2;
		// e = 2*(e + 2-d)* (1-d) - 1
		return e//(2*e - 1);
		// return this.noise.hashPos(x, y) & ((1 << 24) -1);
	}
}

class Vec2 {

	constructor(x, y) {
		this.x = x;
		this.y = y;
	}

	hash() {
		return this.x + "," + this.y;
	}

	surface() {
		return this.x * this.y;
	}

	length() {
		return Math.hypot(this.x, this.y);
	}

	normalize() {
		return this.mult(1/this.length());
	}

	mult(n) {
		return vec2(this.x * n, this.y * n);
	}

	add(v) {
		return vec2(this.x + v.x, this.y + v.y);
	}
}

Vec2.unHash = function Vec2UnHash(str) {
	let [x, y] = str.split(",").map(i => i | 0)
	return new Vec2(x, y);
}

function vec2(x, y) {
	return new Vec2(x, y);
}

class Vec2Queue {
	constructor(width, height) {
		this.heap = new PriorityQueue();
		// this.map = new Map();
	}
	insert(pos, height) {
		// let h = pos.hash();
		// if (this.known.has(h)){
			// return;
		// }
		this.heap.add(height, pos);
		// this.map.set(pos.hash(), height);
	}
	pop() {
		let [height, pos] = this.heap.remove();
		return [pos, height];
		// let smallest = null;
		// let lowest = Infinity;
		// this.map.forEach((height, pos) => {
		// 	if (height < lowest) {
		// 		smallest = pos;
		// 		lowest = height;
		// 	}
		// });
		// this.map.delete(smallest);
		// return [Vec2.unHash(smallest), lowest];
	}
	size() {
		return this.heap.heap.length
		// return this.map.size;
	}
}

function nrm(x) {
	return x / Math.hypot(1, x);
}

class RiverGen {
	constructor(seed, width, height) {
		this.seed = seed;
		this.width = width;
		this.height = height;
		this.size = vec2(width, height);
		let mapgen = new MapGen(seed, width, height)
		this.map = new Float32Array(width * height);
		this.originalHeight = new Float32Array(width * height);
		for (let y=0; y<this.size.y; ++y) {
			for (let x=0; x<this.size.x; ++x) {
				let h = mapgen.heightAt(x, y)
				this.map[x + y*this.size.x] = h;
				this.originalHeight[x + y*this.size.x] = h;
			}
		}
		this.wetness = new Uint8ClampedArray(width * height);
		this.erosion = 0
		// this.erode()
	}

	normal(x, y) {
		const scale = 60;
		return vec2(
			nrm(scale * (this.heightAt(x-1, y) - this.heightAt(x, y))) + nrm(scale * (this.heightAt(x, y) - this.heightAt(x+1, y))),
			nrm(scale * (this.heightAt(x, y-1) - this.heightAt(x, y))) + nrm(scale * (this.heightAt(x, y) - this.heightAt(x, y+1)))
		).mult(0.25);
		// let d1 = this.heightAt(x-1, y-1) - this.heightAt(x+1, y+1);
		// let d2 = this.heightAt(x-1, y+1
		// return vec2(this.heightAt(x-1, y) - this.heightAt(x+1, y), this.heightAt(x, y-1) - this.heightAt(x, y+1)).mult(12);
	}

	blur(center, sides, corners) {
		sides = sides || 0;
		corners = corners || 0;
		for (let y=1; y<this.size.y-1; ++y) {
			for (let x=1; x<this.size.x-1; ++x) {
				let w = this.size.x;
				let i = x + y * w;
				this.map[i] =
					center * this.map[i]
					+ sides * (this.map[i+1] + this.map[i-1] + this.map[i+w] + this.map[i-w]) / 4
					+ corners * (this.map[i+1+w] + this.map[i-1+w] + this.map[i+1-w] + this.map[i-1-w]) / 4;
			}
		}
	}


	erode(steps){
		const dt = 1.2;
		const density = 1;
		const evapRate = 0.001;
		const depositionRate = 0.1;
		const minVol = 0.01;
		const friction = 0.005;
		for (let i=0; i<steps;++i) {
			// let fx = Math.abs(randomize(randomize(this.seed ^ randomize(randomize(i*3)*101)))) % this.width;
			// let fy = Math.abs(randomize(randomize(randomize(this.seed*7) ^ randomize(randomize(i*13)*41)))) % this.height;
			let fx = (Math.random() * this.width);
			let fy = (Math.random() * this.height);

			// console.log(x, y);
			// let dx = 0;
			// let dy = 0;
			let speed = vec2(0, 0);
			let volume = 1;
			let sediment = 0;
			while (volume > minVol) {
				let x = fx | 0;
				let y = fy | 0
				if (x < 1 || y < 1 || x >= this.with-2 || y >= this.height-2) {
					break;
				}
				let h = this.heightAt(x, y);
				// if (h < -0.3) {
				// if (h < -0.9) {
					// break;
				// }
				// let norm = normal(x, y);
				// dx += (this.heightAt(x-1, y) - this.heightAt(x+1, y))*0.5;
				// dy += (this.heightAt(x, y-1) - this.heightAt(x, y+1))*0.5;
				speed = speed.add(this.normal(x, y).mult(1/(volume * density)));;
				fx += speed.x;
				fy += speed.y;
				speed = speed.mult(1 - dt * friction);
				// dx *= 0.9;
				// dy *= 0.9;
				if (fx < 1 || fy < 1 || fx >= this.size.x-2 || fy >= this.size.y-2) {
					break;
				}
				let maxsediment = Math.max(0, volume * speed.length() * (h - this.heightAt(fx|0, fy|0))*1);
				let sdiff = maxsediment - sediment;
				sediment += dt * depositionRate * sdiff;
				// console.log(maxsediment);
				this.map[x + y*this.width] -= dt * volume * depositionRate * sdiff;
				volume *= (1 - dt * evapRate);
			}

			// let concentration = 0;
			// for (let speed=100; speed--;) {
				// this.map[x + y*this.size.x] -= (speed / 50 - 1);
			// }
		}
		this.erosion += steps;
	}

	addRivers(count){
		for (let i=0; i<count; ++i) {
			let x = Math.random() * this.width|0;//(randomize(this.seed ^ randomize(seed*73))&0xffffff) % this.width;
			let y = Math.random() * this.height|0;//(randomize(randomize(this.seed*5) ^ randomize(seed*37))&0xffffff) % this.height;
			// console.log(x, y, this.mapgen.heightAt(x, y));
			let fringe = new Vec2Queue();
			let known = new Uint8Array(this.width * this.height);
			fringe.insert(vec2(x, y), this.heightAt(x, y));
			known[x + y * this.width] = 1;
			let limit=100000;
			let lastHeight = -Infinity;
			while (fringe.size() && --limit) {
				let [pos, height] = fringe.pop();
				if (height < 0) {
					break;
				}
				if (height < lastHeight) {
					while (height < lastHeight && height >= 0) {
						known[pos.x + pos.y * this.width] = 1;
						this.wetness[pos.x + pos.y*this.width] += 1;
						let dh = 0;
						lastHeight = height;
						let neighbours = [
							[pos.x - 1, pos.y, 1],
							[pos.x + 1, pos.y, 1],
							[pos.x, pos.y - 1, 1],
							[pos.x, pos.y + 1, 1],
							[pos.x - 1, pos.y - 1, 0.707],
							[pos.x + 1, pos.y - 1, 0.707],
							[pos.x - 1, pos.y + 1, 0.707],
							[pos.x + 1, pos.y + 1, 0.707],
						];
						for (let [nx, ny, f] of neighbours) {
						// for (let nx = pos.x-1; nx <= pos.x+1; ++nx) {
							// for (let ny = pos.y-1; ny <= pos.y+1; ++ny) {
								if (known[nx + this.width * ny]) {
									continue;
								}
								let nh = this.heightAt(nx, ny);
								let score = (lastHeight - nh) * f;
								if (score > dh) {
									height = nh;
									dh = score;
									pos = vec2(nx, ny);
								}
							// }
						}
					}
					fringe.insert(pos, height);
					continue;
				}
				this.wetness[pos.x + pos.y*this.width] += 1;
				for (let nx = pos.x-1; nx <= pos.x+1; ++nx) {
					for (let ny = pos.y-1; ny <= pos.y+1; ++ny) {
						if (known[nx + this.width * ny]) {
							continue;
						}
						known[nx + this.width * ny] = 1;
						fringe.insert(vec2(nx, ny), this.heightAt(nx, ny));
					}
				}
				lastHeight = height;
			}
		}
	}

	heightAt(x, y) {
		return this.map[x + y*this.size.x];
	}


	colorAt(x, y) {
		let ind = x + y * this.size.x;
		let h = this.map[ind];
		let g = Math.min(1, Math.max(0, 0.2 + h - this.originalHeight[ind]))
		let c = h*256 | ((g * 255) << 8)
		let w = this.wetness[ind];
		if (w > 3) {
			return c  | (Math.min(255, 150 + w)<<16);
		} else {
			return c;
		}
	}
}

class World {
	constructor(size, seed) {
		this.size = size;
		this.seed = seed;
		this.gen = new RiverGen(seed, size.x, size.y);
	}
	draw(id) {
		if (!id) {
			id = "worldlevel";
		}
		let canvas = document.getElementById(id);
		canvas.width = this.size.x;
		canvas.height = this.size.y;
		let ctx = canvas.getContext("2d");
		let drawData = new ArrayBuffer(this.size.surface() * 4);
		let pixel32Array = new Uint32Array(drawData)
		let imgDataArray = new Uint8ClampedArray(drawData);
		for (let x=0; x<this.size.x; ++x) {
			for (let y=0; y<this.size.y; ++y) {
				pixel32Array[x + y*this.size.x] = this.gen.colorAt(x, y);
			}
		}
		for (let i=0; i<this.size.surface(); ++i) {
			imgDataArray[i*4 + 3] = 0xff;
		}
		ctx.putImageData(new ImageData(imgDataArray, this.size.x, this.size.y),0,0);
	}
}

const maxErosion = 5000

function main() {
	let startTime = Date.now()
	let size = 1024;
	let world = new World(vec2(size, size), 12);
	world.draw();
	world.draw("original");
	window.world = world;
	// console.log("generated world", (Date.now() - startTime) / 1000);
	this.requestAnimationFrame(() => stepErosion(world));
	// requestAnimationFrame(() => step(world, 10000));
	// for (let i=0;i<10; ++i) {
	// 	world.gen.addRiver(i);
	// 	world.draw();
	// }
}

function stepErosion(world) {
	if (world.gen.erosion > maxErosion ){
		requestAnimationFrame(() => step(world));
		return;
	}
	let startTime = Date.now();
	world.gen.erode(1000);
	world.gen.blur(0.99, 0.008, 0.002);
	console.log((Date.now() - startTime) / 1000);
	world.draw();
	requestAnimationFrame(()=>stepErosion(world));
}


function step(world) {


	let startTime = Date.now()
	// for (let i=0; i<1000; ++i) {
		world.gen.addRivers(1000);
	// }
	let riverTime = Date.now();
	world.draw();
	console.log("river", (riverTime - startTime) / 1000, "draw", (Date.now() - riverTime) / 1000);
	requestAnimationFrame(() => step(world));
}

window.addEventListener("load", main);
