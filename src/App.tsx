import { Outlet } from 'react-router-dom'
import './App.css'
import { Sidebar } from './Sidebar.tsx'

function App() {
  return (
    <div className="appContainer">
      <Sidebar/>
      <div className="content">
        <Outlet />
        </div>
    </div>
  )
}

export default App


