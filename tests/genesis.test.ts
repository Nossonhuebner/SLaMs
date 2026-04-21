import { describe, it, expect } from 'vitest'
import { genesisChars, genesisClean } from '../src/util/genesis'

describe('genesis corpus', () => {
    it('raw genesisChars is non-empty', () => {
        expect(genesisChars.length).toBeGreaterThan(10_000)
    })

    it('genesisClean contains only lowercase a-z and spaces', () => {
        // Scan the whole corpus; if any char is outside the allowed set, fail with its codepoint.
        const offender = [...genesisClean].find((c) => {
            const cp = c.charCodeAt(0)
            const isLower = cp >= 'a'.charCodeAt(0) && cp <= 'z'.charCodeAt(0)
            const isSpace = c === ' '
            return !(isLower || isSpace)
        })
        expect(offender).toBeUndefined()
    })

    it('genesisClean preserves approximately the same length as raw (punctuation-only stripping)', () => {
        // Raw has punctuation + apostrophes + the occasional digit. Clean is ~90%+ of raw.
        expect(genesisClean.length).toBeGreaterThan(genesisChars.length * 0.85)
    })
})
