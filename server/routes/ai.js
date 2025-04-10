// routes/ai.js or similar
const express = require("express");
const router = express.Router();
const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

router.post("/ai-health-chat", async (req, res) => {
  const { prompt } = req.body;

  if (!prompt.toLowerCase().includes("health") && !prompt.toLowerCase().includes("doctor")) {
    return res.json({
      msg: "I'm here to help with health-related questions only. Please ask something related to health or wellness.",
    });
  }

  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const chat = model.startChat({
      systemInstruction:
        "You are a helpful AI health assistant. Only respond to health-related questions.",
    });

    const result = await chat.sendMessage(prompt);
    const response = await result.response;
    return res.json({ msg: response.text() });
  } catch (err) {
    console.error("Gemini error:", err);
    res.status(500).json({ msg: "AI Assistant is currently unavailable." });
  }
});

module.exports = router;
