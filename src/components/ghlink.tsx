import ghlogo from '../assets/ghlogo.png';
import {Link} from "@mui/material";

function GhLink({repositoryLink}: {repositoryLink: string}) {
    return (
        <Link href={repositoryLink} target="_blank" rel="noopener" style={{margin: '0 auto', width: 'fit-content'}}>
            <img src={ghlogo} style={{width: '50px'}}/>
        </Link>
    )
}

export default GhLink;