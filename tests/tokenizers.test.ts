import { describe, it, expect } from 'vitest'
import { CharTokenizerLite, WordTokenizerLite } from '../src/util/tokenizers'

describe('CharTokenizerLite', () => {
    const tok = new CharTokenizerLite()

    it('has a fixed 28-char vocab: *, a-z, period', () => {
        expect(tok.vocab).toEqual('*abcdefghijklmnopqrstuvwxyz.'.split(''))
        expect(tok.vocab).toHaveLength(28)
    })

    it('treats * as the special token at index 0', () => {
        expect(tok.specialToken).toBe(0)
        expect(tok.vocab[tok.specialToken]).toBe('*')
    })

    it('encodes a string of vocab chars to defined indices', () => {
        const encoded = tok.encode('hello')
        expect(encoded).toEqual([
            'abcdefghijklmnopqrstuvwxyz'.indexOf('h') + 1,
            'abcdefghijklmnopqrstuvwxyz'.indexOf('e') + 1,
            'abcdefghijklmnopqrstuvwxyz'.indexOf('l') + 1,
            'abcdefghijklmnopqrstuvwxyz'.indexOf('l') + 1,
            'abcdefghijklmnopqrstuvwxyz'.indexOf('o') + 1,
        ])
    })

    it('encode/decode roundtrip is stable', () => {
        const s = 'the.quick.brown.fox'
        expect(tok.decode(tok.encode(s))).toBe(s)
    })

    it('also accepts a pre-split string[] input to encode', () => {
        const s = 'cat'
        expect(tok.encode(s.split(''))).toEqual(tok.encode(s))
    })
})

describe('WordTokenizerLite', () => {
    const corpus = 'the*quick*brown*fox.the*lazy*dog.'
    const tok = new WordTokenizerLite(corpus)

    it('splits on * and treats . as its own word', () => {
        expect(tok.vocab).toContain('the')
        expect(tok.vocab).toContain('quick')
        expect(tok.vocab).toContain('.')
    })

    it('specialToken is the index of .', () => {
        expect(tok.vocab[tok.specialToken]).toBe('.')
    })

    it('encode then decode yields space-joined words', () => {
        const encoded = tok.encode('the*quick*fox')
        const decoded = tok.decode(encoded)
        expect(decoded).toBe('the quick fox')
    })
})
