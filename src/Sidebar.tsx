import Drawer from '@mui/material/Drawer';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import { Link } from 'react-router-dom';
import GhLink from './components/ghlink'

const drawerWidth = 240;

export function Sidebar() {
  return (
    <Drawer
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
      variant="permanent"
      anchor="left"
    >
      <List>
          <ListItem disablePadding>
            <ListItemButton component={Link} to={`/`}>
              <ListItemText primary="Home" />
            </ListItemButton>
          </ListItem>

          <ListItem disablePadding>
            <ListItemButton component={Link} to={`ngrams`}>
              <ListItemText primary="N-grams" />
            </ListItemButton>
          </ListItem>

          <ListItem disablePadding>
            <ListItemButton component={Link} to={`mlp`}>
              <ListItemText primary="Multi-Layer Perceptron" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton component={Link} to={`mlp-v2`}>
              <ListItemText primary="Multi-Layer Perceptron V2 (tf)" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton component={Link} to={`wavenet`}>
              <ListItemText primary="WaveNet" />
            </ListItemButton>
          </ListItem>
          <ListItem disablePadding>
            <ListItemButton component={Link} to={`transformer`}>
              <ListItemText primary="Transformer" />
            </ListItemButton>
          </ListItem>
      </List>
      <GhLink repositoryLink="https://github.com/Nossonhuebner/SLaMs"/>
    </Drawer>
  );
}