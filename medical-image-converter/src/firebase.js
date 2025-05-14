// src/firebase.js
import { initializeApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';

// Firebaseの設定情報（Firebase Consoleから取得）
const firebaseConfig = {
    apiKey: "AIzaSyCJDxVSoZZkDRj-MB_b4WivlhzuhBN030M",
    authDomain: "fyp-g16.firebaseapp.com",
    projectId: "fyp-g16",
    storageBucket: "fyp-g16.firebasestorage.app",
    messagingSenderId: "1053010482218",
    appId: "1:1053010482218:web:733c66afa9223d6ca91f53",
    measurementId: "G-H1C49L9EW7"//opt
  };

// Firebaseの初期化
const app = initializeApp(firebaseConfig);

// 認証サービスの取得
export const auth = getAuth(app);
export default app;