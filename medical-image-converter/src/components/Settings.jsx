import React, { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNavigate } from 'react-router-dom';
import { updatePassword, updateProfile } from 'firebase/auth';
import { ArrowLeft, User, Lock, Sun, Moon } from 'lucide-react';

const Settings = () => {
  const { currentUser, updateEmail } = useAuth();
  const [displayName, setDisplayName] = useState(currentUser?.displayName || '');
  const [email, setEmail] = useState(currentUser?.email || '');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleBack = () => navigate('/');

  const handleProfileUpdate = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');
    setLoading(true);

    try {
      const updates = [];
      
      if (displayName !== currentUser.displayName) {
        await updateProfile(currentUser, { displayName });
      }
      
      if (email !== currentUser.email) {
        updates.push(updateEmail(email));
      }
      
      await Promise.all(updates);
      setMessage('Profile updated successfully');
    } catch (err) {
      setError(`Failed to update profile: ${err.message}`);
    }
    
    setLoading(false);
  };

  const handlePasswordUpdate = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');

    if (newPassword !== confirmPassword) {
      return setError('Passwords do not match');
    }

    if (newPassword.length < 6) {
      return setError('Password should be at least 6 characters');
    }

    setLoading(true);
    try {
      await updatePassword(currentUser, newPassword);
      setMessage('Password updated successfully');
      setNewPassword('');
      setConfirmPassword('');
    } catch (err) {
      if (err.code === 'auth/requires-recent-login') {
        setError('For security, please log out and back in before changing password');
      } else {
        setError(`Failed to update password: ${err.message}`);
      }
    }
    setLoading(false);
  };

  const [isDark, setIsDark] = useState(() => {
    return localStorage.getItem('theme') === 'dark';
  });

  const toggleTheme = () => {
    const html = document.documentElement;
    const newTheme = isDark ? 'light' : 'dark';
    html.classList.toggle('dark', newTheme === 'dark');
    localStorage.setItem('theme', newTheme);
    setIsDark(newTheme === 'dark');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm">
        <div className="max-w-4xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <button 
                onClick={handleBack}
                className="mr-4 p-1 rounded-full hover:bg-gray-100"
              >
                <ArrowLeft size={24} className="text-gray-600" />
              </button>
              <h1 className="text-2xl font-bold text-gray-900">Account Settings</h1>
            </div>
            <button
              onClick={toggleTheme}
              className="flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50"
            >
              {isDark ? (
                <>
                  <Moon size={16} className="mr-2" />
                  Dark Mode
                </>
              ) : (
                <>
                  <Sun size={16} className="mr-2" />
                  Light Mode
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-4xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {/* Alerts */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-400">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {message && (
          <div className="mb-6 p-4 bg-green-50 border-l-4 border-green-400">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-green-700">{message}</p>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Profile Information */}
          <div className="bg-white shadow-sm rounded-lg overflow-hidden border border-gray-200">
            <div className="px-4 py-5 sm:px-6 bg-gray-50 border-b border-gray-200">
              <div className="flex items-center">
                <User size={20} className="text-gray-500 mr-2" />
                <h2 className="text-lg font-medium text-gray-900">Profile Information</h2>
              </div>
            </div>
            <div className="px-4 py-5 sm:p-6">
              <form onSubmit={handleProfileUpdate}>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Display Name</label>
                  <input
                    type="text"
                    value={displayName}
                    onChange={(e) => setDisplayName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  Update Profile
                </button>
              </form>
            </div>
          </div>

          {/* Change Password */}
          <div className="bg-white shadow-sm rounded-lg overflow-hidden border border-gray-200">
            <div className="px-4 py-5 sm:px-6 bg-gray-50 border-b border-gray-200">
              <div className="flex items-center">
                <Lock size={20} className="text-gray-500 mr-2" />
                <h2 className="text-lg font-medium text-gray-900">Change Password</h2>
              </div>
            </div>
            <div className="px-4 py-5 sm:p-6">
              <form onSubmit={handlePasswordUpdate}>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">New Password</label>
                  <input
                    type="password"
                    value={newPassword}
                    onChange={(e) => setNewPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-700 mb-1">Confirm Password</label>
                  <input
                    type="password"
                    value={confirmPassword}
                    onChange={(e) => setConfirmPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <button
                  type="submit"
                  disabled={loading}
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50"
                >
                  Change Password
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;