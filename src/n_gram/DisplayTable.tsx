type Props = {
    grid: number[][];
    intToChar: string[];
}
function DisplayTable({ grid, intToChar }: Props){
    const max = Math.max(...grid.flat());
    const intensity = (x: number) => Math.floor(x / max * 255);
    return (
        <table>
            <thead>
                <tr>
                    <th>char</th>
                    {/* column headers */}
                    {grid[0].map((_, i) => <th key={i}>{intToChar[i]}</th>)}
                </tr>
            </thead>
            <tbody>
                {grid.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                        <td>{intToChar[rowIdx]}</td>
                        {grid[0].map((_, colIdx) => (
                            <td key={`${colIdx}-${rowIdx}`}>
                                <div style={{backgroundColor: `rgb(0,${intensity(row[colIdx])},0)`}}>
                                    <label>{intToChar[rowIdx] + intToChar[colIdx]}</label>
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
