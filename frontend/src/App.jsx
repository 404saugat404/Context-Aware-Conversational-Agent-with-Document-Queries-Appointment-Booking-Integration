import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatPage from './pages/ChatPage'
import AppointmentPage from './pages/AppointmentPage'
import IngestPage from './pages/IngestPage'
import './App.css'

function App() {
  const [page, setPage] = useState('chat')

  const renderPage = () => {
    switch (page) {
      case 'chat':        return <ChatPage />
      case 'appointment': return <AppointmentPage />
      case 'ingest':      return <IngestPage />
      default:            return <ChatPage />
    }
  }

  return (
    <div className="app-layout">
      <Sidebar activePage={page} onNavigate={setPage} />
      <main className="main-content">
        {renderPage()}
      </main>
    </div>
  )
}

export default App
