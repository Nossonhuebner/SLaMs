import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import {
  createBrowserRouter,
  RouterProvider,
} from "react-router-dom";
import ErrorPage from "./error-page";
import HomePage from './Home.tsx';
import { SlammyGrammy } from './components/n_gram/slammy-grammy.tsx';

const router = createBrowserRouter([
  {
    path: "/SLaMs",
    element: <App />,
    errorElement: <ErrorPage />,
    children: [
    {
      path: "/SLaMs/", index: true, element: <HomePage/>
    },
    {
      path: "/SLaMs/ngrams/",element: <SlammyGrammy />
    }
  ]
  }
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
