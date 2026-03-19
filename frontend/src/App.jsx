import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import AppointmentPage from './pages/AppointmentPage'
import './App.css'

function App() {
  const [page, setPage] = useState('chat')

  return (
    <div className="app-layout">
      <Sidebar activePage={page} onNavigate={setPage} />
      <main className="main-content">
        {page === 'chat' ? <ChatPage /> : <AppointmentPage />}
      </main>
    </div>
  )
}

export default App
