const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

async function generateGradients() {
  const outputDir = path.join(__dirname, 'assets');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  // Generate main gradient
  await sharp({
    create: {
      width: 720,
      height: 405,
      channels: 4,
      background: {
        r: 28,
        g: 40,
        b: 83,
        alpha: 1
      }
    }
  })
  .linearGradient([
    { stop: 0, color: '#1C2833' },  // Deep navy
    { stop: 1, color: '#2E4053' }   // Slate gray
  ])
  .toFile(path.join(outputDir, 'gradient_bg.png'));

  console.log('✅ Generated gradient_bg.png');
}

generateGradients().catch(console.error);
