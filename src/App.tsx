import React, { useState, useRef, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import { FaLinkedinIn } from "react-icons/fa6";
import { FaGithub } from "react-icons/fa6";
import { FaInstagram } from "react-icons/fa6";

interface Prediction {
  emotion: string;
  scores: number[];
}

const App: React.FC = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const emotions = [
    "Angry 😡",
    "Disgust 🤢",
    "Fear 😰",
    "Happy 😁",
    "Neutral 😐",
    "Sad 😔",
    "Surprise 🤯",
  ];

  useEffect(() => {
    const loadModel = async () => {
      setLoading(true);
      try {
        const loadedModel = await tf.loadLayersModel("/model.json");
        setModel(loadedModel);
      } catch (error) {
        console.error("Error loading the model:", error);
      } finally {
        setLoading(false);
      }
    };
    loadModel();
  }, []);

  useEffect(() => {
    const startWebcam = async () => {
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
          });
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
          }
        }
      } catch (error) {
        console.error("Error accessing the webcam:", error);
      }
    };
    startWebcam();
  }, []);

  const detect = async () => {
    if (model && videoRef.current) {
      try {
        // Capture frame from webcam
        const video = tf.browser.fromPixels(videoRef.current);
        const resized = tf.image.resizeBilinear(video, [128, 128]);
        const grayscale = tf.image.rgbToGrayscale(resized);
        const normalized = grayscale.expandDims(0).div(255.0);
        console.log(normalized);

        const predictions = (await model.predict(normalized)) as tf.Tensor;
        const predictionsArray = await predictions.data();
        // @ts-ignore
        const predictedIndex = predictions.argMax(-1).arraySync()[0];

        const predictedEmotion = emotions[predictedIndex];

        setPrediction({
          emotion: predictedEmotion,
          scores: Array.from(predictionsArray),
        });

        tf.dispose([video, resized, grayscale, normalized, predictions]);
      } catch (error) {
        console.error("Error during prediction:", error);
      }
    }
  };

  // Run detection every 200ms
  useEffect(() => {
    const interval = setInterval(() => detect(), 200);
    return () => clearInterval(interval);
  }, [model]);
  if (loading)
    return (
      <p
        style={{
          position: "absolute",
          top: "50%",
          left: "47%",
          fontSize: 32,
        }}
      >
        Loading...
      </p>
    );
  return (
    <div className="app-container">
      <h1>Emotion Detection</h1>
      <p>By Adetona Adegbite</p>
      <div className="socials">
        <a href="https://www.linkedin.com/in/adetona-adegbite?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app">
          <FaLinkedinIn color="white" size={32} />
        </a>
        <a href="">
          <FaGithub color="white" size={32} />
        </a>
        <a href="https://www.instagram.com/tona_tech?igsh=MTU0em1jMGl5MnJ0aw%3D%3D&utm_source=qr">
          <FaInstagram color="white" size={32} />
        </a>
      </div>
      <div className="video-container">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          width="640"
          height="480"
        />
        <canvas
          ref={canvasRef}
          width="640"
          height="480"
          style={{ display: "none" }}
        ></canvas>
      </div>
      {prediction && (
        <div className="prediction-box">
          <h2>Predictions:</h2>
          <p>
            <strong>Emotion:</strong> {prediction.emotion}
          </p>
          <h3>Probabilities:</h3>
          <ul>
            {prediction.scores.map((score, index) => (
              <li key={index}>
                {emotions[index]}: {score.toFixed(2)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default App;
