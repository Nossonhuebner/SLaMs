import { describe, it, expect } from 'vitest'
import { Tokenizer, createDataset } from '../src/util/tok2'

describe('Tokenizer (tok2)', () => {
    it('builds a sorted, unique vocab from the corpus', () => {
        const tok = new Tokenizer('hello world')
        expect(tok.vocab).toEqual(['*', 'd', 'e', 'h', 'l', 'o', 'r', 'w'])
    })

    it('treats the end char (space by default) as the special char internally', () => {
        const tok = new Tokenizer('a b c')
        expect(tok.vocab).toContain('*')
        expect(tok.vocab).not.toContain(' ')
        expect(tok.specialChar).toBe('*')
        expect(tok.endChar).toBe(' ')
    })

    it('specialToken maps to the index of the special char in vocab', () => {
        const tok = new Tokenizer('hello world')
        expect(tok.vocab[tok.specialToken]).toBe('*')
    })

    it('encode/decode roundtrip preserves the original (modulo space handling)', () => {
        const tok = new Tokenizer('hello world')
        expect(tok.decode(tok.encode('hello'))).toBe('hello')
        expect(tok.decode(tok.encode('hello world'))).toBe('hello world')
    })

    it('encode substitutes endChar with specialChar before lookup', () => {
        const tok = new Tokenizer('hello world')
        const encoded = tok.encode(' ')
        expect(encoded).toHaveLength(1)
        expect(encoded[0]).toBe(tok.specialToken)
    })

    it('returns no undefined tokens when encoding chars drawn from the training corpus', () => {
        const corpus = 'abcdefghijklmnop'
        const tok = new Tokenizer(corpus)
        const encoded = tok.encode(corpus)
        expect(encoded.every((n) => typeof n === 'number' && !Number.isNaN(n))).toBe(true)
    })
})

describe('createDataset (tok2)', () => {
    it('produces sliding-window samples, skipping any with undefined tokens', () => {
        // Corpus has no spaces, so the special char isn't in the vocab, so the
        // leading padding tokens come back undefined and get filtered out.
        const tok = new Tokenizer('abcde')
        const dataset = createDataset('abcde', tok, 2)
        expect(dataset.length).toBe(3)
        expect(dataset[0]).toEqual({ input: [0, 1], output: 2 }) // a b → c
    })

    it('each sample has input of length contextLength and a scalar output', () => {
        const tok = new Tokenizer('abcde')
        const dataset = createDataset('abcde', tok, 3)
        for (const item of dataset) {
            expect(item.input).toHaveLength(3)
            expect(typeof item.output).toBe('number')
        }
    })

    it('final sample predicts the last character of the corpus', () => {
        const tok = new Tokenizer('abcde')
        const dataset = createDataset('abcde', tok, 2)
        const last = dataset[dataset.length - 1]
        expect(tok.decode([last.output])).toBe('e')
    })

    it('padding lands in vocab when corpus contains spaces → leading samples use special char', () => {
        // With a space in the corpus, * enters the vocab and padded tokens are valid.
        const tok = new Tokenizer('a b')
        const dataset = createDataset('a b', tok, 2)
        // Padded: "  a b" → after replace: "**a*b" → encoded length 5 → 3 samples.
        expect(dataset.length).toBe(3)
        // First input is all padding (both tokens are the special token).
        expect(dataset[0].input.every((t) => t === tok.specialToken)).toBe(true)
        // Its target is 'a'.
        expect(tok.decode([dataset[0].output])).toBe('a')
    })
})
