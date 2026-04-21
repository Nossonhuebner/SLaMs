import { describe, it, expect } from 'vitest'
import { createDataset } from '../src/util/util'

describe('createDataset (util/util.ts — TF-flavoured)', () => {
    it('yields tokens.length - contextLength samples', () => {
        const tokens = [1, 2, 3, 4, 5]
        const ds = createDataset(tokens, 2)
        expect(ds).toHaveLength(3)
    })

    it('each input is a contiguous slice of length contextLength', () => {
        const tokens = [1, 2, 3, 4, 5]
        const ds = createDataset(tokens, 2)
        expect(ds[0]).toEqual({ input: [1, 2], output: 3 })
        expect(ds[1]).toEqual({ input: [2, 3], output: 4 })
        expect(ds[2]).toEqual({ input: [3, 4], output: 5 })
    })

    it('returns an empty array when tokens.length <= contextLength', () => {
        expect(createDataset([1, 2], 3)).toEqual([])
        expect(createDataset([1, 2, 3], 3)).toEqual([])
    })
})
