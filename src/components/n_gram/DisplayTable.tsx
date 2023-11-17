type Props = {
    grid: number[][];
    decode: (tokens: number[]) => string;
}
function DisplayTable({ grid, decode }: Props){
    const max = Math.max(...grid.flat());
    const intensity = (x: number) => Math.floor(x / max * 255);
    return (
        <table>
            <thead>
                <tr>
                    <th>char</th>
                    {/* column headers */}
                    {grid[0].map((_, i) => <th key={i}>{decode([i])}</th>)}
                </tr>
            </thead>
            <tbody>
                {grid.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                        <td>{decode([rowIdx])
                        
                        }</td>
                        {grid[0].map((_, colIdx) => (
                            <td key={`${colIdx}-${rowIdx}`}>
                                <div style={{backgroundColor: `rgb(0,${intensity(row[colIdx])},0)`, width: '50px'}}>
                                    <label>{decode([rowIdx, colIdx])}</label>
                                    <div>{parseFloat(row[colIdx].toFixed(2))}</div>
                                </div>
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}
export default DisplayTable;
