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
		this.noise.SetFractalOctaves(12);
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

	toUint() {
		return this.x | (this.y << 16);
	}
}

Vec2.fromUint = function Vec2FromUint(uint) {
	return new Vec2(uint & 0xffff, uint >> 16);
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
		this.erosion = 0;
		this.lakeCounter = 0;
		this.lakeIds = new Uint32Array(width * height);
		this.lakeProps = [null]
		this.nextPos = new Uint32Array(width * height);
	}

	direct() {
		this.wetness = new Uint8ClampedArray(this.size.surface());
		this.lakeCounter = 0;
		this.lakeIds = new Uint32Array(this.size.surface());
		this.lakeProps = [null]
		let calcNextTime = Date.now();
		// for (let y=0; y<this.size.y; ++y) {
			// for (let x=0; x<this.size.x; ++x) {
		let ntiles = this.size.surface()
		for (let index=0; index<ntiles; index++) {
				// let height = this.heightAt(x, y);
				let height = this.map[index]
				if (height < 0 || this.lakeIds[index]) {
					continue;
				}
				let x = index % this.size.x;
				let y = (index / this.size.x)|0;
				let neighbours = [
					[x - 1, y, 1],
					[x + 1, y, 1],
					[x, y - 1, 1],
					[x, y + 1, 1],
					[x - 1, y - 1, 0.707],
					[x + 1, y - 1, 0.707],
					[x - 1, y + 1, 0.707],
					[x + 1, y + 1, 0.707],
				];
				let nextPos = null;
				let dh = 0;
				for (let [nx, ny, f] of neighbours) {
					let np = vec2(nx, ny);
					if (this.outOfBounds(np)) {
						continue;
					}
					let nh = this.heightAt(nx, ny);
					let score = (height - nh) * f;
					if (score > dh) {
						dh = score;
						nextPos = np;
					}
				}
				if (nextPos !== null) {
					this.nextPos[x + this.size.x * y] = this.index(nextPos) + 1;
					// this.nextIndex[x + this.size.x * y] = this.index(nextPos) + 1;
				} else {
					this.fillLake(vec2(x, y));
				}
			// }
		}
		console.log("calc next", (Date.now() - calcNextTime) / 1000);
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


	addStream(x, y){
		// let x = Math.random() * this.size.x | 0;
		// let y = Math.random() * this.size.y | 0;
		// let pos = vec2(x, y);
		let index = x + this.size.x * y;
		let height;// = this.getHeight(pos);
		// let limit = 10000
		while ((height = this.map[index]) >= 0) {
			// let index = this.index(pos);
			if (index === 0) {
				return;
			}
			this.wetness[index] += 1;
			let lakeId = this.lakeIds[index];
			if (lakeId !== 0) {
				let lake = this.lakeProps[lakeId];
				lake.wetness += 1;
				index = this.index(lake.exit);
				continue;
			}
			if (this.nextPos[index] !== 0) {
				index = this.nextPos[index] - 1;
				continue;
			}
			console.error("not computed tile", index);
		}
	}

	fillLake(start) {
		if (this.index(start) === null) {
			console.error("trying to fill a lake at invalid positon", start);
			return null;
		}
		// let known = new Uint8Array(this.width * this.height);
		let known = new Set();
		let fringe = new PriorityQueue();
		fringe.add(this.getHeight(start), start);
		// known[this.index(start)] = 1;
		known.add(this.index(start));
		let lakeId = ++this.lakeCounter;
		let lastHeight = -Infinity;
		let lake = {wetness: 1, exit: null, size: 0};//, fringe: fringe, known: known, tiles: new Set()};
		this.lakeProps[lakeId] = lake;
		while (true) {
			let [height, pos] = fringe.remove();
			let index = this.index(pos)
			let oldLakeId = this.lakeIds[index];
			if (oldLakeId === lakeId) {
				continue;
			}
			if (oldLakeId !== 0) {
				let oldLake = this.lakeProps[oldLakeId];
				if (this.lakeIds[this.index(oldLake.exit)] === lakeId) {
					// other lake is flowing into this lake; absorb it
					this.absorbLake(pos, oldLakeId, lakeId, lake, fringe, known);
					continue;
				} else {
					// lake is lower: this lake flows into it
					lake.exit = pos;
					break;
				}
			}
			if (height < lastHeight) {
				lake.exit = pos;
				break
			}
			lake.size += 1;
			lastHeight = height;
			this.lakeIds[index] = lakeId;
			// lake.tiles.add(index);
			for (let nx = pos.x-1; nx <= pos.x+1; ++nx) {
				for (let ny = pos.y-1; ny <= pos.y+1; ++ny) {
					let np = vec2(nx, ny)
					if (known.has(this.index(np))) {
						continue;
					}
					known.add(this.index(np));// = 1;
					fringe.add(this.getHeight(np), np);
				}
			}
		}
		return lake.exit;
	}

	absorbLake(pos, oldLakeId, lakeId, lake, fringe, known) {
		// let oldLake = this.lakeProps[oldLakeId];
		// this.lakeProps[oldLakeId] = null;
		// for (let k of oldLake.known) {
		// 	lake.known.add(k);
		// }
		// for (let [height, pos] of oldLake.fringe.heap) {
		// 	lake.fringe.add(height, pos);
		// }
		// for (let index of oldLake.tiles) {
		// 	this.lakeIds[index] = lakeId;
		// 	lake.tiles.add(index);
		// }
		// return;
		let absorbFringe = [pos];
		while (absorbFringe.length) {
			let pos = absorbFringe.pop();
			let id = this.lakeIds[this.index(pos)]
			if (id === oldLakeId) {
				lake.size += 1;
				this.lakeIds[this.index(pos)] = lakeId;
				for (let nx = pos.x-1; nx <= pos.x+1; ++nx) {
					for (let ny = pos.y-1; ny <= pos.y+1; ++ny) {
						let np = vec2(nx, ny);
						let nind = this.index(np)
						if (known.has(nind) && this.lakeIds[nind] !== oldLakeId) {
							continue;
						}
						known.add(nind);
						// known[nind] = 1;
						absorbFringe.push(np);
					}
				}
			} else if (id !== lakeId) {
				fringe.add(this.getHeight(pos), pos);
			}
		}
		this.lakeProps[oldLakeId] = null;
	}

	addRivers(count){
		// for (let i=0; i<count; ++i) {
		// 	let x = Math.random() * this.size.x | 0;
		// 	let y = Math.random() * this.size.y | 0;
		// 	this.addStream(x, y);
		// }
		for (let y=0; y<this.size.y; y++){
			for (let x=0; x<this.size.x; x++){
				this.addStream(x, y);
			}
		}

	}

	outOfBounds(pos) {
		return pos.x < 0 || pos.y < 0 || pos.x >= this.size.x || pos.y >= this.size.y;
	}

	index(pos) {
		if (pos.x < 0 || pos.y < 0 || pos.x >= this.size.x || pos.y >= this.size.y) {
			return null;
		}
		return pos.x + this.size.x * pos.y;
	}

	fromIndex(index) {
		return vec2(index % this.size.x, (index / this.size.x)|0);
	}

	getHeight(pos) {
		return this.map[pos.x + this.size.x * pos.y];
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
		let lake = this.lakeProps[this.lakeIds[ind]];
		if (lake) {
			w = lake.wetness;
		}
		if (w > 150) {
			return c  | (Math.min(255,  w)<<16);
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
	let size = 2048;
	let world = new World(vec2(size, size), 12);
	world.draw();
	world.draw("original");
	window.world = world;
	world.gen.direct();
	// console.log("generated world", (Date.now() - startTime) / 1000);
	this.requestAnimationFrame(() => step(world));
	setInterval(() => world.draw(), 200);
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
	// world.draw();
	requestAnimationFrame(()=>stepErosion(world));
}


function step(world) {


	world.gen.direct();
	let directTime = Date.now();
	let ntiles = world.size.surface();
	let order = new BigInt64Array(ntiles);
	for (let i=0; i<ntiles; i++) {
		order[i] = BigInt((10 - world.gen.map[i]) * 1e6 | 0) * 0x100000000n + BigInt(i);
	}
	order.sort();
	let startTime = Date.now();
	console.log("sort", (startTime - directTime) / 1000);
	// for (let i=0; i<1000; ++i) {
		world.gen.addRivers(100000);
	// }
	let riverTime = Date.now();
	// world.draw();
	console.log("river", (riverTime - startTime) / 1000);
	requestAnimationFrame(() => step(world));
}

window.addEventListener("load", main);
