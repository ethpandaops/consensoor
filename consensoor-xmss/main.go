package main

/*
#include <stdint.h>
*/
import "C"

import (
	"encoding/binary"
	"sync"
	"unsafe"

	leanxmss "github.com/nalepae/go-leanxmss"
	"golang.org/x/crypto/sha3"
)

const (
	seedSize      = 32
	publicKeySize = 32
	messageSize   = 32
	signatureSize = 2404
	vrfTreeDepth  = 31
	vrfLeafDepth  = 20
	vrfChunkDepth = 10
	vrfLeafCount  = 1 << vrfLeafDepth
	vrfProofSize  = (vrfTreeDepth + 1) * 32
)

var (
	keysMu sync.RWMutex
	keys   = map[[seedSize]byte]*leanxmss.PrivateKey{}
)

func privateKey(seed *C.uint8_t) *leanxmss.PrivateKey {
	var s [seedSize]byte
	copy(s[:], unsafe.Slice((*byte)(unsafe.Pointer(seed)), seedSize))

	keysMu.RLock()
	key, ok := keys[s]
	keysMu.RUnlock()

	if ok {
		return key
	}

	key = leanxmss.NewKeyFromSeed(s)

	keysMu.Lock()
	keys[s] = key
	keysMu.Unlock()

	return key
}

//export xmss_public_key
func xmss_public_key(seed *C.uint8_t, publicKeyOut *C.uint8_t) {
	copy(unsafe.Slice((*byte)(unsafe.Pointer(publicKeyOut)), publicKeySize), privateKey(seed).PublicKey().Bytes())
}

//export xmss_prepare
func xmss_prepare(seed *C.uint8_t, leafIndex C.uint32_t) C.int {
	if err := privateKey(seed).Prepare(uint32(leafIndex)); err != nil {
		return 1
	}

	return 0
}

//export xmss_sign
func xmss_sign(seed *C.uint8_t, leafIndex C.uint32_t, message *C.uint8_t, signatureOut *C.uint8_t) C.int {
	var m [messageSize]byte
	copy(m[:], unsafe.Slice((*byte)(unsafe.Pointer(message)), messageSize))

	sig, err := privateKey(seed).Sign(uint32(leafIndex), m)
	if err != nil {
		return 1
	}

	copy(unsafe.Slice((*byte)(unsafe.Pointer(signatureOut)), signatureSize), sig.Bytes())

	return 0
}

//export xmss_verify
func xmss_verify(publicKey *C.uint8_t, leafIndex C.uint32_t, message *C.uint8_t, signature *C.uint8_t) C.int {
	pk, err := leanxmss.PublicKeyFromBytes(unsafe.Slice((*byte)(unsafe.Pointer(publicKey)), publicKeySize))
	if err != nil {
		return 0
	}

	sig, err := leanxmss.SignatureFromBytes(unsafe.Slice((*byte)(unsafe.Pointer(signature)), signatureSize))
	if err != nil {
		return 0
	}

	var m [messageSize]byte
	copy(m[:], unsafe.Slice((*byte)(unsafe.Pointer(message)), messageSize))

	if leanxmss.Verify(pk, uint32(leafIndex), m, sig) {
		return 1
	}

	return 0
}

type vrfTree struct {
	seed   [seedSize]byte
	levels [vrfLeafDepth - vrfChunkDepth + 1][][32]byte
}

var (
	vrfMu    sync.Mutex
	vrfTrees = map[[seedSize]byte]*vrfTree{}
	vrfZeros = func() (zeros [vrfTreeDepth + 1][32]byte) {
		for i := 1; i <= vrfTreeDepth; i++ {
			zeros[i] = hashPair(&zeros[i-1], &zeros[i-1])
		}

		return zeros
	}()
)

func hashPair(left, right *[32]byte) [32]byte {
	var buf [64]byte
	copy(buf[:32], left[:])
	copy(buf[32:], right[:])

	return keccak256(buf[:])
}

func keccak256(data []byte) [32]byte {
	var out [32]byte

	h := sha3.NewLegacyKeccak256()
	h.Write(data)
	h.Sum(out[:0])

	return out
}

func vrfChunk(seed *[seedSize]byte, chunk uint64) [vrfChunkDepth + 1][][32]byte {
	var levels [vrfChunkDepth + 1][][32]byte

	leaves := make([][32]byte, 1<<vrfChunkDepth)

	var buf [seedSize + 8]byte
	copy(buf[:seedSize], seed[:])

	for i := range leaves {
		binary.LittleEndian.PutUint64(buf[seedSize:], chunk<<vrfChunkDepth+uint64(i))
		leaves[i] = keccak256(buf[:])
	}

	levels[0] = leaves

	for level := 1; level <= vrfChunkDepth; level++ {
		below := levels[level-1]
		nodes := make([][32]byte, len(below)/2)

		for i := range nodes {
			nodes[i] = hashPair(&below[2*i], &below[2*i+1])
		}

		levels[level] = nodes
	}

	return levels
}

func buildVRFTree(seed [seedSize]byte) *vrfTree {
	tree := &vrfTree{seed: seed}
	roots := make([][32]byte, 1<<(vrfLeafDepth-vrfChunkDepth))

	for chunk := range roots {
		roots[chunk] = vrfChunk(&seed, uint64(chunk))[vrfChunkDepth][0]
	}

	tree.levels[0] = roots

	for level := 1; level < len(tree.levels); level++ {
		below := tree.levels[level-1]
		nodes := make([][32]byte, len(below)/2)

		for i := range nodes {
			nodes[i] = hashPair(&below[2*i], &below[2*i+1])
		}

		tree.levels[level] = nodes
	}

	return tree
}

func vrfTreeOf(seed *C.uint8_t) *vrfTree {
	var s [seedSize]byte
	copy(s[:], unsafe.Slice((*byte)(unsafe.Pointer(seed)), seedSize))

	vrfMu.Lock()
	tree, ok := vrfTrees[s]
	vrfMu.Unlock()

	if ok {
		return tree
	}

	tree = buildVRFTree(s)

	vrfMu.Lock()
	vrfTrees[s] = tree
	vrfMu.Unlock()

	return tree
}

func (t *vrfTree) subtreeRoot() [32]byte {
	return t.levels[len(t.levels)-1][0]
}

func (t *vrfTree) root() [32]byte {
	node := t.subtreeRoot()
	for level := vrfLeafDepth; level < vrfTreeDepth; level++ {
		node = hashPair(&node, &vrfZeros[level])
	}

	return node
}

func (t *vrfTree) proof(slot uint64) [vrfTreeDepth + 1][32]byte {
	var proof [vrfTreeDepth + 1][32]byte

	inside := slot < vrfLeafCount

	var chunk [vrfChunkDepth + 1][][32]byte
	if inside {
		chunk = vrfChunk(&t.seed, slot>>vrfChunkDepth)
		proof[0] = chunk[0][slot&(1<<vrfChunkDepth-1)]
	}

	for level := 0; level < vrfTreeDepth; level++ {
		sibling := (slot >> level) ^ 1

		switch {
		case inside && level < vrfChunkDepth:
			proof[level+1] = chunk[level][sibling&(uint64(len(chunk[level]))-1)]
		case inside && level < vrfLeafDepth:
			proof[level+1] = t.levels[level-vrfChunkDepth][sibling]
		case sibling == 0 && level >= vrfLeafDepth:
			node := t.subtreeRoot()
			for l := vrfLeafDepth; l < level; l++ {
				node = hashPair(&node, &vrfZeros[l])
			}

			proof[level+1] = node
		default:
			proof[level+1] = vrfZeros[level]
		}
	}

	return proof
}

//export vrf_root
func vrf_root(seed *C.uint8_t, rootOut *C.uint8_t) {
	root := vrfTreeOf(seed).root()
	copy(unsafe.Slice((*byte)(unsafe.Pointer(rootOut)), 32), root[:])
}

//export vrf_proof
func vrf_proof(seed *C.uint8_t, slot C.uint64_t, proofOut *C.uint8_t) {
	proof := vrfTreeOf(seed).proof(uint64(slot))
	out := unsafe.Slice((*byte)(unsafe.Pointer(proofOut)), vrfProofSize)

	for i := range proof {
		copy(out[i*32:], proof[i][:])
	}
}

func main() {}
