import { useState, useRef, useEffect, useMemo } from 'react';
import English from '../../../raw_english'
import { CharacterTokenizer, clean } from '../../util';
import * as tf from '@tensorflow/tfjs';
import DisplayTable from './DisplayTable';
import { buildGrid, calculateLoss, generateWords } from './util';
import { Select, MenuItem, Button, Switch, FormControlLabel } from '@mui/material'

export function SlammyGrammy() {
    const cleaned = useMemo(() => clean(English), []);
    const [n, setN] = useState<number>(1)
    const [generatedWords, setGeneratedWords] = useState<string>('');
    const [displayedWords, setDisplayedWords] = useState<string>('');
    const [isGenerating, setIsGenerating] = useState(false);
    const tkn = useMemo(() => new CharacterTokenizer(cleaned), [cleaned]);
    const outputRef = useRef<HTMLDivElement>(null);
    const [showNormalized, setShowNormalized] = useState(false);

    const tokens = useMemo(() => tkn.tokenize(cleaned), [tkn, cleaned]);
    const x = useMemo(() => tokens.slice(0, tokens.length - 1), [tokens]);
    const y = useMemo(() => tokens.slice(1, tokens.length), [tokens]);

    const grid = useMemo(() => buildGrid(x, y, tkn), [x, y, tkn]);
    const normalizedGrid = useMemo(() => {
        const grid = buildGrid(x, y, tkn, true)
        console.log(calculateLoss(grid, x, y, tkn))
        return grid
    }, [x, y, tkn]);

    window.tkn = tkn;
    window.txt = cleaned;
    window.tensor = grid;
    window.tf = tf;


    const handleGenerateWords = () => {
        if (isGenerating) return;
        setIsGenerating(true);
        const words = generateWords(normalizedGrid, tkn, 30);
        setGeneratedWords(prev => prev + ' ' + words.join(' '));
    };

    useEffect(() => {
        if (generatedWords.length > displayedWords.length) {
            const timer = setTimeout(() => {
                setDisplayedWords(generatedWords.slice(0, displayedWords.length + 1));
            }, 10); // Adjust this value to change typing speed
            return () => clearTimeout(timer);
        } else {
            setIsGenerating(false);
        }
    }, [generatedWords, displayedWords]);

    useEffect(() => {
        if (outputRef.current) {
            outputRef.current.scrollTop = outputRef.current.scrollHeight;
        }
    }, [displayedWords]);

    return (
        <div>
            <h1>Slam Grams</h1>
            <Select value={n} onChange={(e) => setN(e.target.value as number)}>
                <MenuItem value={1}>1</MenuItem>
                <MenuItem value={2}>2</MenuItem>
            </Select>
            {tkn && (
                <>
                    <Button 
                        variant="contained" 
                        onClick={handleGenerateWords} 
                        style={{ margin: '10px 0' }}
                        disabled={isGenerating}
                    >
                        Generate Words
                    </Button>
                    <div 
                        ref={outputRef}
                        style={{ 
                            height: '100px', 
                            overflow: 'auto', 
                            backgroundColor: '#f8f8f8',
                            padding: '10px',
                            marginBottom: '20px',
                            color: '#333',
                            fontWeight: 500,
                            border: '1px solid #ddd',
                            borderRadius: '4px',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word'
                        }}
                    >
                        {displayedWords}
                    </div>
                    <div style={{ marginBottom: '20px' }}>
                        <FormControlLabel
                            control={
                                <Switch
                                    checked={showNormalized}
                                    onChange={(e) => setShowNormalized(e.target.checked)}
                                />
                            }
                            label="Show normalized probabilities"
                        />
                    </div>
                    <DisplayTable 
                        grid={showNormalized ? normalizedGrid : grid} 
                        decode={(tokens) => tkn.decode(tokens)} 
                    />
                </>
            )}
        </div>
    );
}

declare global {
    interface Window {
        tkn: CharacterTokenizer;
        txt: string;
        tensor: number[][];
        tf: typeof tf;
    }
}