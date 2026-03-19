import './Sidebar.css'

function Sidebar({ activePage, onNavigate }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <span className="brand-icon">+</span>
        <span className="brand-text">HealthBot</span>
      </div>

      <nav className="sidebar-nav">
        <button
          className={`nav-btn ${activePage === 'chat' ? 'active' : ''}`}
          onClick={() => onNavigate('chat')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          Chat
        </button>
        <button
          className={`nav-btn ${activePage === 'appointment' ? 'active' : ''}`}
          onClick={() => onNavigate('appointment')}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
          Book Appointment
        </button>
      </nav>

      <div className="sidebar-footer">
        <span className="footer-text">Conversational Agent v1.0</span>
      </div>
    </aside>
  )
}

export default Sidebar
