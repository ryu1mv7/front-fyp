import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import PrivateRoute from './components/PrivateRoute';
import Login from './components/Login';
import Signup from './components/Signup';
import ForgotPassword from './components/ForgotPassword';
import MedicalImageConverter from './components/MedicalImageConverter';
import Settings from './components/Settings';
import Landing from './components/Landing';

function AppRoutes() {
  const { currentUser } = useAuth();

  return (
    <Routes>
      {/* If user is logged in, redirect '/' to /converter */}
      <Route
        path="/"
        element={
          currentUser ? <Navigate to="/converter" /> : <Landing />
        }
      />
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route
        path="/settings"
        element={
          <PrivateRoute>
            <Settings />
          </PrivateRoute>
        }
      />
      <Route
        path="/converter"
        element={
          <PrivateRoute>
            <MedicalImageConverter />
          </PrivateRoute>
        }
      />
    </Routes>
  );
}

function App() {
  return (
    <Router>
      <AuthProvider>
        <AppRoutes />
      </AuthProvider>
    </Router>
  );
}

export default App;
