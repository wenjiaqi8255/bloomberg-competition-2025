const pptxgen = require('pptxgenjs');
const html2pptx = require('/Users/wenjiaqi/.claude/plugins/cache/claude-scientific-skills/scientific-skills/d17d74dc5d73/scientific-skills/document-skills/pptx/scripts/html2pptx.js');
const fs = require('fs');
const path = require('path');

// Presentation configuration
const PRESENTATION_CONFIG = {
  title: 'Bloomberg Competition Defense Presentation',
  author: 'Quantitative Trading Team',
  layout: 'LAYOUT_16x9'
};

// Color palette - Professional Financial Theme
const COLORS = {
  primary: '1C2833',    // Deep navy
  secondary: '2E4053',  // Slate gray
  accent: '5EA8A7',     // Teal
  highlight: 'EED6D3',  // Light cream
  text: 'F4F6F6',        // Off-white
  success: '40695B',    // Forest green
  warning: 'E07A5F',    // Terracotta
  danger: 'C0392B',     // Red
  dark: '191A19',       // Black
  chart1: '4472C4',     // Blue
  chart2: 'FF6B9D',     // Pink
  chart3: 'F39C12'      // Orange
};

async function createPresentation() {
  const pptx = new pptxgen();
  pptx.layout = PRESENTATION_CONFIG.layout;
  pptx.author = PRESENTATION_CONFIG.author;
  pptx.title = PRESENTATION_CONFIG.title;

  const slidesDir = path.join(__dirname, 'slides');
  const htmlFiles = fs.readdirSync(slidesDir)
    .filter(f => f.endsWith('.html'))
    .sort();

  console.log(`Found ${htmlFiles.length} slide files to process...`);

  for (const htmlFile of htmlFiles) {
    const htmlPath = path.join(slidesDir, htmlFile);
    console.log(`Processing ${htmlFile}...`);

    try {
      const { slide, placeholders } = await html2pptx(htmlPath, pptx);

      // Add charts to placeholder areas based on slide type
      if (htmlFile.includes('slide3_results')) {
        await addResultsChart(slide, placeholders);
      } else if (htmlFile.includes('slide4_comparison')) {
        await addComparisonChart(slide, placeholders);
      } else if (htmlFile.includes('slide5_ablation')) {
        await addAblationChart(slide, placeholders);
      } else if (htmlFile.includes('slide10_metrics')) {
        await addMetricsTable(slide, placeholders);
      }

    } catch (error) {
      console.error(`Error processing ${htmlFile}:`, error.message);
      throw error;
    }
  }

  // Save presentation
  const outputPath = path.join(__dirname, 'defense_presentation.pptx');
  await pptx.writeFile({ fileName: outputPath });
  console.log(`\n✅ Presentation created successfully: ${outputPath}`);

  // Generate thumbnails for visual validation
  console.log('\nGenerating thumbnails for validation...');
  const { spawn } = require('child_process');

  await new Promise((resolve, reject) => {
    const proc = spawn('python', [
      '/Users/wenjiaqi/.claude/plugins/cache/claude-scientific-skills/scientific-skills/document-skills/pptx/scripts/thumbnail.py',
      outputPath,
      path.join(__dirname, 'thumbnails'),
      '--cols', '4'
    ]);

    proc.on('close', (code) => {
      if (code === 0) {
        console.log('✅ Thumbnails generated successfully');
        resolve();
      } else {
        reject(new Error(`Thumbnail generation failed with code ${code}`));
      }
    });
  });
}

async function addResultsChart(slide, placeholders) {
  if (placeholders.length === 0) return;

  const data = [{
    name: 'FF5 Strategy',
    labels: ['Total Return', 'Annualized', 'Sharpe', 'Max Drawdown'],
    values: [40.42, 74.90, 1.17, -66.88]
  }];

  slide.addChart(pptxgen.charts.BAR, data, {
    ...placeholders[0],
    barDir: 'col',
    showTitle: true,
    title: 'FF5 Strategy Key Performance Metrics',
    showLegend: false,
    showCatAxisTitle: false,
    showValAxisTitle: true,
    valAxisTitle: 'Value (%)',
    chartColors: [COLORS.chart1],
    dataLabelPosition: 'outEnd',
    dataLabelColor: 'FFFFFF'
  });
}

async function addComparisonChart(slide, placeholders) {
  if (placeholders.length === 0) return;

  const data = [
    {
      name: 'Without Filter',
      labels: ['Total Return', 'Sharpe', 'Max Drawdown'],
      values: [11.17, 0.62, -73.27]
    },
    {
      name: 'With Filter',
      labels: ['Total Return', 'Sharpe', 'Max Drawdown'],
      values: [40.42, 1.17, -66.88]
    }
  ];

  slide.addChart(pptxgen.charts.BAR, data, {
    ...placeholders[0],
    barGrouping: 'clustered',
    barDir: 'col',
    showTitle: true,
    title: 'Alpha T-Statistic Filtering Impact',
    showLegend: true,
    legendPos: 'b',
    showCatAxisTitle: false,
    showValAxisTitle: true,
    valAxisTitle: 'Value (%)',
    chartColors: [COLORS.chart2, COLORS.chart1]
  });
}

async function addAblationChart(slide, placeholders) {
  if (placeholders.length === 0) return;

  const data = [{
    name: 'Performance Improvement',
    labels: ['Total Return', 'Sharpe Ratio', 'Max Drawdown'],
    values: [262, 89, 8.7]
  }];

  slide.addChart(pptxgen.charts.BAR, data, {
    ...placeholders[0],
    barDir: 'col',
    showTitle: true,
    title: 'Ablation Study: Filtering Impact (%)',
    showLegend: false,
    showCatAxisTitle: false,
    showValAxisTitle: true,
    valAxisTitle: 'Improvement (%)',
    chartColors: [COLORS.success],
    valAxisMinVal: 0,
    dataLabelPosition: 'outEnd',
    dataLabelColor: 'FFFFFF'
  });
}

async function addMetricsTable(slide, placeholders) {
  if (placeholders.length === 0) return;

  const tableData = [
    [
      { text: 'Metric', options: { fill: { color: COLORS.primary }, color: 'FFFFFF', bold: true } },
      { text: 'FF5 Strategy', options: { fill: { color: COLORS.primary }, color: 'FFFFFF', bold: true } },
      { text: 'ML Strategy', options: { fill: { color: COLORS.primary }, color: 'FFFFFF', bold: true } }
    ],
    ['Total Return', '40.42%', '-39.61%'],
    ['Annualized Return', '74.90%', '-35.10%'],
    ['Sharpe Ratio', '1.17', '-0.545'],
    ['Max Drawdown', '-66.88%', '-57.75%'],
    ['Volatility', '90.06%', '52.24%'],
    ['Win Rate', '48.37%', '56.80%']
  ];

  slide.addTable(tableData, {
    ...placeholders[0],
    colW: [3, 3, 3],
    border: { pt: 1, color: 'CCCCCC' },
    align: 'center',
    fontSize: 14
  });
}

// Run the presentation builder
createPresentation().catch(console.error);
