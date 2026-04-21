import { describe, it, expect } from 'vitest'
import { clean, CharacterTokenizer } from '../src/util'

describe('clean', () => {
    it('lowercases English, strips punctuation, maps spaces to *', () => {
        expect(clean('Hello, World!')).toBe('hello*world')
    })

    it('collapses multiple spaces', () => {
        expect(clean('foo   bar')).toBe('foo*bar')
    })

    it('strips the <VV>/<PP> style markup that ships with raw_english', () => {
        expect(clean('<VV>in the beginning')).toBe('in*the*beginning')
    })

    it('detects Hebrew via regex and routes to the Hebrew cleaner', () => {
        const out = clean('בראשית')
        // Hebrew path wraps in <S>...<E> markers.
        expect(out.startsWith('<S>')).toBe(true)
        expect(out.endsWith('<E>')).toBe(true)
    })
})

describe('CharacterTokenizer', () => {
    it('builds a vocab of unique chars from the input', () => {
        const tok = new CharacterTokenizer('aabbcc')
        expect(new Set(tok.vocabulary)).toEqual(new Set(['a', 'b', 'c']))
    })

    it('tokenizes <X> and {Y} as multi-char tokens', () => {
        const tok = new CharacterTokenizer('<S>hi<E>')
        expect(tok.tokenize('<S>hi<E>')).toEqual(['<S>', 'h', 'i', '<E>'])
    })

    it('encode/decode roundtrip preserves the string', () => {
        const tok = new CharacterTokenizer('hello')
        expect(tok.decode(tok.encode('hello'))).toBe('hello')
    })

    it('counts occurrences of each token', () => {
        const tok = new CharacterTokenizer('aaabbc')
        expect(tok.counts['a']).toBe(3)
        expect(tok.counts['b']).toBe(2)
        expect(tok.counts['c']).toBe(1)
    })
})
