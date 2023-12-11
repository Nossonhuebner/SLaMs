import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'
import {
  createHashRouter,
  RouterProvider,
} from "react-router-dom";
import ErrorPage from "./error-page";
import HomePage from './Home.tsx';
import { SlammyGrammy } from './components/n_gram/slammy-grammy.tsx';
import MLP from './components/mlp/mlp.tsx';
import TfMlp from './components/mlp_v2/tf_mlp.tsx';

const router = createHashRouter([
  {
    path: "/",
    element: <App />,
    errorElement: <ErrorPage />,
    children: [
    {
      path: "/", index: true, element: <HomePage/>
    },
    {
      path: "ngrams", element: <SlammyGrammy />
    },
    {
      path: "mlp", element: <MLP/>
    },
    {
      path: "mlp-v2", element: <TfMlp/>
    }
  ]
  }
]);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>,
)
