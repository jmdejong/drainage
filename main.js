"use strict";


const PRIME_A = 134053;
const PRIME_B = 200183;
const PRIME_C = 145931;
const PRIME_D = 161387;
// const MASK = (1 << 48) - 1;

function clamp(num, min, max) {
	return Math.max(min, Math.min(max, num));
}


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
		let ntiles = this.size.surface()
		for (let index=0; index<ntiles; index++) {
			let height = this.map[index]
			if (this.lakeIds[index]) {
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
				let ind = nx + this.size.x * ny;
				let nh = this.map[ind];
				let score = (height - nh ) * f;
				if (score > dh) {
					dh = score;
					nextPos = np;
				}
			}
			if (nextPos !== null) {
				this.nextPos[x + this.size.x * y] = this.index(nextPos) + 1;
			}
		}
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
		let index = x + this.size.x * y;
		length = 0;
		while (index >= this.size.x && index < this.size.x * (this.size.y - 1) && (index + 1) % this.size.x > 1) {
			let height = this.map[index];
			if (index === 0) {
				return;
			}
			let lakeId = this.lakeIds[index];
			if (lakeId !== 0) {
				let lake = this.lakeProps[lakeId];
				if (lake.isSea) {
					return;
				}
				lake.wetness += 1;
				index = this.index(lake.exit);
			} else {
				this.wetness[index] += 1;
				let oldIndex = index;
				index = this.nextPos[index] - 1;
				if (index < 0) {
					console.error("invalid next stream", index, this.fromIndex(oldIndex), length);
					return;
				}
			}
			++length
		}
	}

	fillLake(start) {
		if (this.index(start) === null) {
			console.error("trying to fill a lake at invalid positon", start);
			return null;
		}
		let known = new Set();
		let fringe = new PriorityQueue();
		fringe.add(this.getHeight(start), start);
		known.add(this.index(start));
		let lakeId = ++this.lakeCounter;
		let lake = {wetness: 1, exit: null, size: 0, height: -Infinity};
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
				if (!oldLake) {
					console.error("old lake", oldLake, oldLakeId, pos);
				}
				if (this.lakeIds[this.index(oldLake.exit)] === lakeId) {
					// other lake is flowing into this lake; absorb it
					this.absorbLake(index, oldLakeId, lakeId, lake, fringe, known);
					continue;
				} else {
					// lake is lower: this lake flows into it
					lake.exit = pos;
					break;
				}
			}
			if (height < lake.height) {
				lake.exit = pos;
				break
			}
			lake.size += 1;
			lake.height = height;
			this.lakeIds[index] = lakeId;
			// lake.tiles.add(index);
			for (let nind of this.neighbourIndices(index)) {
				if (known.has(nind)) {
					continue;
				}
				known.add(nind);
				fringe.add(this.map[nind], this.fromIndex(nind));
			}
		}
		return lake.exit;
	}

	absorbLake(ind, oldLakeId, lakeId, lake, fringe, known) {
		let absorbFringe = [ind];
		this.lakeIds[ind] = lakeId;
		lake.size += 1;
		while (absorbFringe.length) {
			let ind = absorbFringe.pop();
			for (let nind of this.neighbourIndices(ind)) {
				let nid = this.lakeIds[nind];
				if (nid === oldLakeId) {
					lake.size += 1;
					this.lakeIds[nind] = lakeId;
					known.add(nind);
					absorbFringe.push(nind)
				} else if (!known.has(nind) && nid !== lakeId) {
					known.add(nind);
					fringe.add(this.map[nind], this.fromIndex(nind));
				}
			}
			// let id = this.lakeIds[ind];
			// if (id === oldLakeId) {
			// 	lake.size += 1;
			// 	this.lakeIds[ind] = lakeId;
			// 	for (let nind of this.neighbourIndices(ind)) {
			// 		if (known.has(nind) && this.lakeIds[nind] !== oldLakeId) {
			// 			continue;
			// 		}
			// 		known.add(nind);
			// 		absorbFringe.push(nind);
			// 	}
			// } else if (id !== lakeId) {
			// 	fringe.add(this.map[ind], this.fromIndex(ind));
			// }
		}
		this.lakeProps[oldLakeId] = null;
	}

	addRivers(){
		for (let y=1; y<this.size.y-1; y++){
			for (let x=1; x<this.size.x-1; x++){
				this.addStream(x, y);
			}
		}

	}

	fillLakes(order) {
		this.lakeCounter = 1;
		this.lakeIds = new Uint32Array(this.size.surface());
		const seaHeight = -0.2;
		let sea = {wetness: 1e9, exit: vec2(-1, -1), size: 0, height: seaHeight, isSea: true}
		this.lakeProps = [null, sea];
		let seaId = 1;
		let seaFringe = [];
		let edges = new Set();
		for (let x=0; x<this.size.x; ++x) {
			edges.add(x);
			edges.add(x + (this.size.y - 1) * this.size.x);
		}
		for (let y=0; y<this.size.y; ++y) {
			edges.add(y*this.size.x);
			edges.add((y+1)*this.size.x - 1);
		}
		for (let edge of edges) {
			if (this.map[edge] < seaHeight) {
				seaFringe.push(edge)
				this.lakeIds[edge] = seaId;
			}
		}
		while (seaFringe.length) {
			let ind = seaFringe.pop();
			sea.size++;
			for (let nind of this.neighbourIndices(ind)) {
				if (this.lakeIds[nind] === 0 && this.map[nind] < seaHeight) {
					this.lakeIds[nind] = seaId;
					seaFringe.push(nind);
				}
			}
		}

		for (let y=1; y<this.size.y-1; ++y) {
			let iy = y * this.size.x;
			for (let x=1; x<this.size.x-1; ++x) {
				let ind = x + iy;
				if (this.lakeIds[ind] !== 0){
					continue;
				}
				let h = this.map[ind];
				let isWell = true;
				for (let nind of this.neighbourIndices(ind)) {
					if (this.map[nind] < h) {
						isWell = false;
						break;
					}
				}
				if (isWell) {
					this.fillLake(vec2(x, y));
				}
			}
		}
	}

	neighbourIndices(ind) {
		if (ind < this.size.x) {
			if (ind === 0) {
				return [this.size.x + 1, 1, this.size.x];
			} else if (ind === this.size.x - 1) {
				return [ind - 1 + this.size.x, ind + this.size.x, ind - 1];
			} else {
				return [ind - 1 + this.size.x, ind + 1 + this.size.x, ind + this.size.x, ind - 1, ind + 1];
			}
		} else if (ind >= this.size.x * (this.size.y - 1)) {
			if (ind === this.size.x * (this.size.y - 1)) {
				return [ind - this.size.x + 1, ind - this.size.x, ind + 1];
			} else if (ind === this.size.x * this.size.y - 1) {
				return [ind - 1 - this.size.x, ind - 1, ind - this.size.x];
			} else {
				return [ind - 1 - this.size.x, ind + 1 - this.size.x, ind - 1, ind + 1, ind - this.size.x];
			}
		} else if (ind % this.size.x === 0) {
			return [ind + 1 - this.size.x, ind + 1 + this.size.x, ind + 1, ind - this.size.x, ind + this.size.x];
		} else if (ind % this.size.x === this.size.x - 1) {
			return [ind - 1 - this.size.x, ind - 1 + this.size.x, ind - 1, ind - this.size.x, ind + this.size.x];
		} else {
			return [ind - 1 - this.size.x, ind + 1 - this.size.x, ind - 1 + this.size.x, ind + 1 + this.size.x, ind - 1, ind + 1, ind - this.size.x, ind + this.size.x];
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
		let w = this.wetness[ind];
		let lake = this.lakeProps[this.lakeIds[ind]];
		if (lake) {
			w = lake.wetness;
			// if (w > 150) {
			// 	h -= lake.height;
			// }
		}
		let r = clamp(h*2, 0, 1);
		let g = clamp(Math.min(h+0.8, 1.8 - h*1.5), 0, 1);//Math.max(0.7-Math.abs(h), 0)//Math.min(1, Math.max(0, 0.2 + h - this.originalHeight[ind]))
		let c = r*255 | ((g * 255) << 8)
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
	let size = 1024;
	let world = new World(vec2(size, size), 12);
	world.draw();
	world.draw("original");
	window.world = world;
	// world.gen.direct();
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

function time(description, fn) {
	let startTime = Date.now();
	let ret = fn();
	let endTime = Date.now();
	console.log(description, (endTime - startTime) / 1000);
	return ret;
}

function step(world) {

	// let lakeStart = Date.now();
	time("filling lakes", () => world.gen.fillLakes());
	// let directStart = Date.now();
	time("calculating slopes", () => world.gen.direct());
	// let sortStart = Date.now();
	// let ntiles = world.size.surface();
	// let order = new BigInt64Array(ntiles);
	// for (let i=0; i<ntiles; i++) {
	// 	order[i] = BigInt((10 - world.gen.map[i]) * 1e6 | 0) * 0x100000000n + BigInt(i);
	// }
	// order.sort();
	// console.log("sort", (startTime - directTime) / 1000);
	// let riverStart = Date.now()
	time("adding rivers", () => world.gen.addRivers());
	// let endTime = Date.now();
	// world.draw();
	// console.log("river", (endTime - riverStart) / 1000);
	// requestAnimationFrame(() => step(world));
}

window.addEventListener("load", main);
