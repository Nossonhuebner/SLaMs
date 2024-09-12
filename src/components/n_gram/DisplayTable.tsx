import './DisplayTable.css'; // We'll create this file for styles

type Props = {
    grid: number[][];
    decode: (tokens: number[]) => string;
}

function DisplayTable({ grid, decode }: Props) {
    const max = Math.max(...grid.flat());
    const intensity = (x: number) => Math.floor((x / max) * 255);

    return (
        <table className="display-table">
            <thead>
                <tr className="table-header">
                    <th className="header-cell">char</th>
                    {grid[0].map((_, i) => (
                        <th key={i} className="header-cell">{decode([i])}</th>
                    ))}
                </tr>
            </thead>
            <tbody>
                {grid.map((row, rowIdx) => (
                    <tr key={rowIdx} className="table-row">
                        <th className="row-header">{decode([rowIdx])}</th>
                        {row.map((cell, colIdx) => (
                            <td
                                key={`${rowIdx}-${colIdx}`}
                                className="table-cell"
                                style={{
                                    backgroundColor: `rgba(0, ${intensity(cell)}, 0, 0.8)`,
                                }}
                            >
                                <div className="cell-content">{decode([rowIdx, colIdx])}</div>
                                <div className="cell-value">{cell.toFixed(2)}</div>
                            </td>
                        ))}
                    </tr>
                ))}
            </tbody>
        </table>
    );
}

export default DisplayTable;
