// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyDG8oglbTfWmdBpfYeK0sGZqPSlg5-DVco",
    authDomain: "dm-ai-tutor.firebaseapp.com",
    projectId: "dm-ai-tutor",
    storageBucket: "dm-ai-tutor.firebasestorage.app",
    messagingSenderId: "2791894921",
    appId: "1:2791894921:web:116afd0df9cc9be632fb58",
    measurementId: "G-D9ZWSKQTTZ",
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
