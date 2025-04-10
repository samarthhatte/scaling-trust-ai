const express = require("express");
const router = express.Router();
const fetch = require("node-fetch");
require("dotenv").config();

router.post("/ask", async (req, res) => {
  const { prompt } = req.body;

  try {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${process.env.GEMINI_API_KEY}`;

    const payload = {
      contents: [
        {
          parts: [{ text: prompt }],
        },
      ],
    };

    const apiRes = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await apiRes.json();
    const message = data?.candidates?.[0]?.content?.parts?.[0]?.text || "No response from Gemini";

    res.json({ message });
  } catch (error) {
    console.error("Gemini Error:", error);
    res.status(500).json({ message: "Server error calling Gemini" });
  }
});

module.exports = router;
