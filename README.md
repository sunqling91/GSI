# GSI
This project implements seasonal phenology date simulations based on the growing season index (GSI) models. These models are specifically designed to analyze and predict phenological transitions in the unique grassland ecosystems of the Qinghai-Tibetan Plateau(QTP). By incorporating key meteorological and environmental factors, the models quantify seasonal growth conditions and provide spatially resolved phenology maps.
**1. AGSI Model:**
The AGSI model integrates the following factors:
Maximum temperature (Tmax)
Minimum temperature (Tmin)
Photoperiod (day length)
Precipitation (Pre)
**2. GSI Model:**
The GSI model incorporates:
Minimum temperature (Tmin)
Vapor pressure deficit (VPD)
Photoperiod (day length)
Multi-year Data Processing: Efficiently processes daily meteorological data across multiple years with flexible input formats.
Dynamic Parameter Calculation: Computes critical indices like temperature thresholds, water availability, and photoperiod constraints for plant growth.
Phenology Date Extraction: Identifies spring and autumn phenological dates using rolling average smoothing and threshold-based detection.
Spatial Analysis Outputs: Generates spatially resolved phenology maps in GeoTIFF format for use in GIS tools and ecological analysis.
**Key Features:**
Tailored to QTP Grasslands: Both AGSI and GSI models are applied exclusively to the grassland ecosystems of the QTP, accounting for its unique climatic and ecological conditions.
Multi-year Data Processing: Efficiently processes daily meteorological data across multiple years with support for flexible input formats.
Dynamic Parameter Calculation: Computes critical indices such as temperature thresholds, water availability, and photoperiod constraints for grassland growth.
Phenology Date Extraction: Identifies spring and autumn phenological dates using rolling average smoothing and threshold-based detection.
Spatial Analysis Outputs: Generates high-resolution phenology maps in GeoTIFF format, compatible with GIS tools for visualization and further analysis.
**Applications:**
Monitoring vegetation dynamics and growth trends in the QTP.
Assessing the impact of climate change on grassland ecosystems.
Supporting agricultural and ecological resource management specific to the QTP.
**Technical Details:**
Programming Language: Python 3.x
Dependencies: NumPy, Pandas, Rasterio, tqdm
Output Formats: GeoTIFF files for spatial visualization and GIS compatibility
