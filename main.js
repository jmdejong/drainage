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

class TileQueue {
	constructor(keyfunc) {
		this.heap = new PriorityQueue(keyfunc);
		this.known = new Set();
	}
	add(ind) {
		if (!this.known.has(ind)) {
			this.heap.add(ind);
			this.known.add(ind);
		}
	}
	remove() {
		return this.heap.remove();
	}
	size() {
		return this.heap.heap.length;
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
		this.wetness = new Int32Array(width * height);
		this.erosion = 0;
		this.lakeCounter = 0;
		this.lakeIds = new Uint32Array(width * height);
		this.lakeProps = [null]
		this.nextPos = new Uint32Array(width * height);
		this.errTiles = new Uint8ClampedArray(width * height);
	}

	direct() {
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
		while (!this.isEdge(index)) {
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
				index = lake.exit;
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
		let fringe = new TileQueue(i => this.map[i]);
		fringe.add(this.index(start));
		let lakeId = ++this.lakeCounter;
		let lake = {wetness: 0, exit: null, size: 0, height: -Infinity};
		this.lakeProps[lakeId] = lake;
		while (true) {
			let index = fringe.remove();
			let oldLakeId = this.lakeIds[index];
			if (oldLakeId === lakeId) {
				continue;
			}
			if (oldLakeId !== 0) {
				let oldLake = this.lakeProps[oldLakeId];
				if (!oldLake) {
					console.error("old lake", oldLake, oldLakeId, this.fromIndex(index));
				}
				if (oldLake.height < lake.height) {
					// lake is lower: this lake flows into it
					lake.exit = index;
					break;
				} else {
					// other lake is flowing into this lake; absorb it
					this.absorbLake(index, oldLakeId, lakeId, lake, fringe);
					lake.height = oldLake.height
					continue;
				}
			}
			let height = this.map[index];
			if (height < lake.height) {
				lake.exit = index;
				if (oldLakeId !== 0) {
					let oldLake = this.lakeProps[oldLakeId];
					if (oldLake.height >= lake.height){
						console.error("lake draining into higher lake wtf");
						console.error(lakeId, lake, this.fromIndex(lake.exit));
						console.error(oldLakeId, oldLake, this.fromIndex(oldLake.exit));
						for (let x=-2; x<=2; ++x) {
							for (let y=-2; y<=2; ++y) {
								let p = vec2(x, y).add(this.fromIndex(lake.exit))
								console.log(p, this.lakeIds[this.index(p)], this.surfaceHeight(this.index(p)));
							}
						}
					}
				}
				break
			}
			lake.size += 1;
			lake.height = height;
			this.lakeIds[index] = lakeId;
			for (let nind of this.neighbourIndices(index)) {
				fringe.add(nind);
			}
		}
		return lake.exit;
	}

	absorbLake(ind, oldLakeId, lakeId, lake, fringe) {
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
					absorbFringe.push(nind)
				} else if (nid !== lakeId) {
					fringe.add(nind);
				}
			}
		}
		this.lakeProps[oldLakeId] = null;
	}

	addRivers(){
		this.wetness = new Int32Array(this.size.surface());
		for (let y=1; y<this.size.y-1; y++){
			for (let x=1; x<this.size.x-1; x++){
				this.addStream(x, y);
			}
		}
	}

	addOrderedRivers(order){
		let isProcessed = new Uint8ClampedArray(this.size.surface());
		this.errTiles = new Uint8ClampedArray(this.size.surface());
		this.wetness = new Int32Array(this.size.surface());
		for (let i=order.length; i--;) {
			// let [height, ind] = this.fromBig(order[i]);
			let ind = order[i];
			let height = this.map[ind];
			isProcessed[ind] = 1;
			if (this.isEdge(ind)) {
				continue;
			}
			this.wetness[ind]++;
			if (this.lakeIds[ind] !== 0) {
				let lake = this.lakeProps[this.lakeIds[ind]];
				lake.wetness += this.wetness[ind];
				if (!lake.isSea) {
					let next = lake.exit;
					this.wetness[next] += this.wetness[ind];
					if (isProcessed[next]) {
						console.error("out of order lake", this.fromIndex(ind), this.map[ind], this.surfaceHeight(ind), this.lakeIds[ind], this.fromIndex(next), this.map[next], this.surfaceHeight(next), this.lakeIds[next]);
						this.errTiles[ind] |= 1;
						this.errTiles[next] |= 2;
					}
				}
				continue;
			}
			let neighbours = [
				[ind - 1 - this.size.x, 0.707],
				[ind + 1 - this.size.x, 0.707],
				[ind - 1 + this.size.x, 0.707],
				[ind + 1 + this.size.x, 0.707],
				[ind - 1, 1],
				[ind + 1, 1],
				[ind - this.size.x, 1],
				[ind + this.size.x, 1]
			];
			let next = null;
			let dh = 0;
			for (let [nind, weight] of neighbours) {
				let nh = this.map[nind];
				let score = (height - nh ) * weight;
				if (score > dh) {
					dh = score;
					next = nind;
				}
			}
			if (next === null) {
				console.error("endorheic basin found")
			}
			if (isProcessed[next]) {
				console.error("out of order stream");
			}
			if (this.lakeIds[next] !== 0) {
				let lake = this.lakeProps[this.lakeIds[next]];
				lake.wetness += this.wetness[ind];
				if (!lake.isSea) {
					let exit = lake.exit;
					this.wetness[exit] += this.wetness[ind];
					if (isProcessed[exit]) {
						console.error("out of order lakestream");
					}
				}
			} else {
				this.wetness[next] += this.wetness[ind];
			}
		}
	}

	fillSea() {
		this.lakeCounter = 1;
		this.lakeIds = new Uint32Array(this.size.surface());
		const seaHeight = -0.2;
		let sea = {wetness: 1e9, exit: -999999999, size: 0, height: seaHeight, isSea: true}
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
	}

	fillLakes() {
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

	toBig(height, ind) {
		return BigInt((height + 5) * 1e7 | 0) << 32n | BigInt(ind)
	}

	fromBig(big) {
		return [
			Number(BigInt.asUintN(32, big >> 32n)) / 1e7 - 5,
			Number(BigInt.asUintN(32, big))
		];
	}

	order() {
		let sorted = new Uint32Array(this.size.surface());
		let sh = new Float32Array(this.size.surface())
		for (let i=0; i<sorted.length; ++i) {
			sorted[i] = i;
			sh[i] = this.surfaceHeight(i);
		}
		sorted.sort((a, b) => sh[a] - sh[b]);
		return sorted;
	}

	surfaceHeight(i) {
		if (this.lakeIds[i] !== 0) {
			return this.lakeProps[this.lakeIds[i]].height;
		} else {
			return this.map[i];
		}
	}

	isEdge(index) {
		return index < this.size.x || index >= this.size.x * (this.size.y - 1) || (index + 1) % this.size.x <= 1;
	}

	colorAt(x, y) {
		let ind = x + y * this.size.x;
		if (this.errTiles[ind]) {
			let err = this.errTiles[ind];
			if (err === 1) {
				return 0xff;
			} else {
				return 0xff00ff;
			}
		}
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
		let g = clamp(Math.min(h+0.8, 1.8 - h*1.5), 0, 1);
		let c = r*255 | ((g * 255) << 8)
		if (w > 50) {
			return c  | (Math.min(255,  w+100)<<16);
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



function time(description, fn) {
	let startTime = Date.now();
	let ret = fn();
	let endTime = Date.now();
	console.log(description, (endTime - startTime) / 1000);
	return ret;
}

function main() {
	let size = 1024;
	let world = new World(vec2(size, size), 12);
	world.draw();
	world.draw("original");
	window.world = world;
	this.requestAnimationFrame(() => step(world));
	setInterval(() => world.draw(), 200);
}

const maxErosion = 5000

function stepErosion(world) {
	if (world.gen.erosion > maxErosion ){
		requestAnimationFrame(() => step(world));
		return;
	}
	let startTime = Date.now();
	world.gen.erode(1000);
	world.gen.blur(0.99, 0.008, 0.002);
	console.log((Date.now() - startTime) / 1000);
	requestAnimationFrame(()=>stepErosion(world));
}

function step(world) {

	time("filling sea", () => world.gen.fillSea());
	time("filling lakes", () => world.gen.fillLakes());
	// time("calculating slopes", () => world.gen.direct());
	// time("adding rivers", () => world.gen.addRivers());
	let order = time("sorting tiles", () => world.gen.order());
	time("adding rivers in order", () => world.gen.addOrderedRivers(order));
}

window.addEventListener("load", main);
