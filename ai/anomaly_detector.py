"""
Anomaly Detector
AI-powered detection of unusual patterns in ocean data
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import sys
sys.path.append('..')
from config import settings
from database.models import Anomaly, AnomalySeverity

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects anomalies in ARGO oceanographic data"""
    
    # Normal ranges for ocean parameters (approximate)
    NORMAL_RANGES = {
        "temperature": {"min": -2, "max": 35, "surface_min": 20, "surface_max": 32},
        "salinity": {"min": 30, "max": 40, "typical_range": (33, 37)},
        "oxygen": {"min": 0, "max": 400, "hypoxic_threshold": 60},
        "chlorophyll": {"min": 0, "max": 50, "bloom_threshold": 10}
    }
    
    # Anomaly type definitions
    ANOMALY_TYPES = {
        "temperature_spike": "Unusual temperature value detected",
        "salinity_outlier": "Salinity outside normal range",
        "temperature_inversion": "Temperature increasing with depth (unusual)",
        "density_anomaly": "Unusual density structure",
        "oxygen_depletion": "Low oxygen zone detected",
        "chlorophyll_bloom": "High chlorophyll indicating algal bloom",
        "sensor_drift": "Possible sensor calibration issue",
        "mixed_layer_anomaly": "Unusual mixed layer depth",
        "profile_shape_anomaly": "Unusual profile shape detected"
    }
    
    def __init__(self, contamination: float = 0.05):
        """
        Args:
            contamination: Expected proportion of anomalies (0.05 = 5%)
        """
        self.contamination = contamination
        
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            self.scaler = StandardScaler()
        else:
            logger.warning("scikit-learn not available, using statistical methods only")
    
    def detect_profile_anomalies(
        self, 
        measurements: List[Dict[str, Any]],
        profile_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a single profile
        
        Args:
            measurements: List of measurement dicts with pressure, temperature, salinity, etc.
            profile_info: Optional profile metadata (lat, lon, date)
        
        Returns:
            List of detected anomalies
        """
        if not measurements:
            return []
        
        anomalies = []
        
        # Convert to arrays
        pressures = np.array([m.get('pressure', 0) for m in measurements])
        temps = np.array([m.get('temperature') for m in measurements], dtype=float)
        sals = np.array([m.get('salinity') for m in measurements], dtype=float)
        
        # Replace None with NaN
        temps = np.where(temps == None, np.nan, temps)
        sals = np.where(sals == None, np.nan, sals)
        
        # 1. Check for out-of-range values
        anomalies.extend(self._check_range_anomalies(measurements, pressures, temps, sals))
        
        # 2. Check for temperature inversions
        anomalies.extend(self._check_temperature_inversions(pressures, temps))
        
        # 3. Check for statistical outliers
        anomalies.extend(self._check_statistical_outliers(measurements, temps, sals))
        
        # 4. Check profile shape
        if len(measurements) > 10:
            anomalies.extend(self._check_profile_shape(pressures, temps, sals))
        
        # 5. Check for sensor drift (if we have adjusted values)
        anomalies.extend(self._check_sensor_drift(measurements))
        
        # 6. Check mixed layer (if surface data available)
        if pressures.min() < 20:
            anomalies.extend(self._check_mixed_layer(pressures, temps, sals))
        
        return anomalies
    
    def _check_range_anomalies(
        self,
        measurements: List[Dict],
        pressures: np.ndarray,
        temps: np.ndarray,
        sals: np.ndarray
    ) -> List[Dict]:
        """Check for values outside physical ranges"""
        anomalies = []
        
        # Temperature checks
        temp_valid = ~np.isnan(temps)
        if temp_valid.any():
            # Below minimum
            too_cold = temps < self.NORMAL_RANGES["temperature"]["min"]
            if too_cold.any():
                idx = np.where(too_cold)[0][0]
                anomalies.append({
                    "type": "temperature_spike",
                    "severity": AnomalySeverity.HIGH,
                    "description": f"Temperature below minimum ({temps[idx]:.2f}°C at {pressures[idx]:.0f}dbar)",
                    "pressure_range": (float(pressures[idx]), float(pressures[idx])),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.95
                })
            
            # Above maximum
            too_warm = temps > self.NORMAL_RANGES["temperature"]["max"]
            if too_warm.any():
                idx = np.where(too_warm)[0][0]
                anomalies.append({
                    "type": "temperature_spike",
                    "severity": AnomalySeverity.HIGH,
                    "description": f"Temperature above maximum ({temps[idx]:.2f}°C at {pressures[idx]:.0f}dbar)",
                    "pressure_range": (float(pressures[idx]), float(pressures[idx])),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.95
                })
        
        # Salinity checks
        sal_valid = ~np.isnan(sals)
        if sal_valid.any():
            sal_low = sals < self.NORMAL_RANGES["salinity"]["min"]
            sal_high = sals > self.NORMAL_RANGES["salinity"]["max"]
            
            if sal_low.any() or sal_high.any():
                outlier_mask = sal_low | sal_high
                idx = np.where(outlier_mask)[0][0]
                anomalies.append({
                    "type": "salinity_outlier",
                    "severity": AnomalySeverity.MEDIUM,
                    "description": f"Salinity outside range ({sals[idx]:.2f} PSU at {pressures[idx]:.0f}dbar)",
                    "pressure_range": (float(pressures[idx]), float(pressures[idx])),
                    "affected_parameters": ["salinity"],
                    "confidence": 0.9
                })
        
        # Check oxygen if available
        for i, m in enumerate(measurements):
            oxy = m.get('oxygen')
            if oxy is not None:
                if oxy < self.NORMAL_RANGES["oxygen"]["hypoxic_threshold"]:
                    anomalies.append({
                        "type": "oxygen_depletion",
                        "severity": AnomalySeverity.MEDIUM,
                        "description": f"Low oxygen zone ({oxy:.1f} μmol/kg at {pressures[i]:.0f}dbar)",
                        "pressure_range": (float(pressures[i]), float(pressures[i])),
                        "affected_parameters": ["oxygen"],
                        "confidence": 0.85
                    })
                    break  # Just report first occurrence
        
        return anomalies
    
    def _check_temperature_inversions(
        self,
        pressures: np.ndarray,
        temps: np.ndarray
    ) -> List[Dict]:
        """Check for temperature inversions (increasing T with depth)"""
        anomalies = []
        
        valid_mask = ~np.isnan(temps)
        if valid_mask.sum() < 5:
            return anomalies
        
        # Calculate temperature gradient
        valid_temps = temps[valid_mask]
        valid_pressures = pressures[valid_mask]
        
        # Normally temperature decreases with depth below mixed layer
        # Check for significant inversions (warming with depth > 0.5°C/100dbar)
        temp_diff = np.diff(valid_temps)
        pressure_diff = np.diff(valid_pressures)
        
        # Avoid division by zero
        pressure_diff = np.where(pressure_diff == 0, 1, pressure_diff)
        gradient = temp_diff / pressure_diff * 100  # per 100 dbar
        
        # Find significant warming with depth (below 200m to exclude mixed layer)
        deep_mask = valid_pressures[:-1] > 200
        if deep_mask.any():
            strong_inversion = gradient[deep_mask] > 0.5
            if strong_inversion.any():
                idx = np.where(deep_mask)[0][np.where(strong_inversion)[0][0]]
                anomalies.append({
                    "type": "temperature_inversion",
                    "severity": AnomalySeverity.LOW,
                    "description": f"Temperature inversion at {valid_pressures[idx]:.0f}dbar",
                    "pressure_range": (float(valid_pressures[idx]), float(valid_pressures[idx+1])),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.7
                })
        
        return anomalies
    
    def _check_statistical_outliers(
        self,
        measurements: List[Dict],
        temps: np.ndarray,
        sals: np.ndarray
    ) -> List[Dict]:
        """Use statistical methods to find outliers"""
        anomalies = []
        
        # Z-score method for temperature
        valid_temps = temps[~np.isnan(temps)]
        if len(valid_temps) >= 10:
            z_scores = np.abs(stats.zscore(valid_temps))
            outliers = z_scores > 3
            
            if outliers.any():
                # Map back to original indices
                valid_indices = np.where(~np.isnan(temps))[0]
                outlier_indices = valid_indices[outliers]
                
                if len(outlier_indices) > 0:
                    idx = outlier_indices[0]
                    anomalies.append({
                        "type": "temperature_spike",
                        "severity": AnomalySeverity.MEDIUM,
                        "description": f"Statistical outlier in temperature (z={z_scores[outliers][0]:.1f})",
                        "pressure_range": (float(measurements[idx].get('pressure', 0)), float(measurements[idx].get('pressure', 0))),
                        "affected_parameters": ["temperature"],
                        "confidence": 0.75,
                        "detection_method": "z_score"
                    })
        
        return anomalies
    
    def _check_profile_shape(
        self,
        pressures: np.ndarray,
        temps: np.ndarray,
        sals: np.ndarray
    ) -> List[Dict]:
        """Check for unusual profile shapes using Isolation Forest"""
        if not SKLEARN_AVAILABLE:
            return []
        
        anomalies = []
        
        # Create features from profile characteristics
        valid_mask = ~np.isnan(temps) & ~np.isnan(sals)
        if valid_mask.sum() < 10:
            return anomalies
        
        valid_temps = temps[valid_mask]
        valid_sals = sals[valid_mask]
        valid_pressures = pressures[valid_mask]
        
        # Calculate profile features
        features = []
        
        # Surface values
        surface_idx = np.argmin(valid_pressures)
        features.append(valid_temps[surface_idx])
        features.append(valid_sals[surface_idx])
        
        # Deep values (if available)
        if valid_pressures.max() > 500:
            deep_mask = valid_pressures > 500
            features.append(np.mean(valid_temps[deep_mask]))
            features.append(np.mean(valid_sals[deep_mask]))
        else:
            features.extend([np.nan, np.nan])
        
        # Gradients
        features.append(np.std(valid_temps))
        features.append(np.std(valid_sals))
        
        # If we have enough features, we could use Isolation Forest
        # For now, just check for unusual surface-deep differences
        if len(features) >= 4 and not np.isnan(features[2]):
            temp_diff = features[0] - features[2]  # Surface - Deep
            # Unusual if surface is colder than deep (outside tropics) or very large difference
            if temp_diff < -2 or temp_diff > 25:
                anomalies.append({
                    "type": "profile_shape_anomaly",
                    "severity": AnomalySeverity.LOW,
                    "description": f"Unusual temperature profile (surface-deep diff: {temp_diff:.1f}°C)",
                    "pressure_range": (float(valid_pressures.min()), float(valid_pressures.max())),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.6
                })
        
        return anomalies
    
    def _check_sensor_drift(self, measurements: List[Dict]) -> List[Dict]:
        """Check for sensor drift by comparing raw and adjusted values"""
        anomalies = []
        
        # Compare temperature with adjusted temperature
        diffs = []
        for m in measurements:
            t = m.get('temperature')
            t_adj = m.get('temperature_adjusted')
            if t is not None and t_adj is not None:
                diffs.append(abs(t - t_adj))
        
        if diffs:
            max_diff = max(diffs)
            mean_diff = np.mean(diffs)
            
            # Large correction suggests sensor drift
            if max_diff > 0.5 or mean_diff > 0.2:
                anomalies.append({
                    "type": "sensor_drift",
                    "severity": AnomalySeverity.LOW,
                    "description": f"Possible sensor drift (max correction: {max_diff:.2f}°C)",
                    "pressure_range": (0, 2000),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.65
                })
        
        return anomalies
    
    def _check_mixed_layer(
        self,
        pressures: np.ndarray,
        temps: np.ndarray,
        sals: np.ndarray
    ) -> List[Dict]:
        """Check for unusual mixed layer depth"""
        anomalies = []
        
        valid_mask = ~np.isnan(temps) & (pressures < 500)
        if valid_mask.sum() < 5:
            return anomalies
        
        valid_temps = temps[valid_mask]
        valid_pressures = pressures[valid_mask]
        
        # Simple MLD estimate: depth where T differs from surface by 0.5°C
        surface_temp = valid_temps[np.argmin(valid_pressures)]
        temp_diff = np.abs(valid_temps - surface_temp)
        
        mld_mask = temp_diff > 0.5
        if mld_mask.any():
            mld = valid_pressures[mld_mask].min()
            
            # Very shallow or very deep MLD might be anomalous
            if mld < 10:
                anomalies.append({
                    "type": "mixed_layer_anomaly",
                    "severity": AnomalySeverity.LOW,
                    "description": f"Very shallow mixed layer ({mld:.0f}m)",
                    "pressure_range": (0, float(mld)),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.6
                })
            elif mld > 200:
                anomalies.append({
                    "type": "mixed_layer_anomaly",
                    "severity": AnomalySeverity.LOW,
                    "description": f"Very deep mixed layer ({mld:.0f}m)",
                    "pressure_range": (0, float(mld)),
                    "affected_parameters": ["temperature"],
                    "confidence": 0.6
                })
        
        return anomalies
    
    def detect_batch_anomalies(
        self,
        profiles: List[Tuple[Dict, List[Dict]]]
    ) -> Dict[int, List[Dict]]:
        """
        Detect anomalies in batch of profiles
        
        Args:
            profiles: List of (profile_info, measurements) tuples
        
        Returns:
            Dict mapping profile index to detected anomalies
        """
        results = {}
        
        for i, (profile_info, measurements) in enumerate(profiles):
            anomalies = self.detect_profile_anomalies(measurements, profile_info)
            if anomalies:
                results[i] = anomalies
        
        return results


def create_anomaly_records(
    profile_id: int,
    anomalies: List[Dict]
) -> List[Anomaly]:
    """Convert anomaly dicts to database records"""
    records = []
    
    for a in anomalies:
        record = Anomaly(
            profile_id=profile_id,
            anomaly_type=a["type"],
            severity=a.get("severity", AnomalySeverity.LOW),
            description=a.get("description", ""),
            affected_parameters=a.get("affected_parameters", []),
            pressure_range_start=a.get("pressure_range", (0, 0))[0],
            pressure_range_end=a.get("pressure_range", (0, 0))[1],
            confidence_score=a.get("confidence", 0.5),
            detection_method=a.get("detection_method", "statistical")
        )
        records.append(record)
    
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with synthetic data
    detector = AnomalyDetector()
    
    # Create test profile with some anomalies
    test_measurements = []
    for p in range(0, 2000, 20):
        t = 28 - 0.008 * p + np.random.normal(0, 0.1)
        s = 35 + 0.001 * p + np.random.normal(0, 0.05)
        
        # Add an anomaly
        if p == 500:
            t = 35  # Temperature spike
        
        test_measurements.append({
            "pressure": p,
            "depth": p * 0.99,
            "temperature": t,
            "salinity": s,
            "temp_qc": 1,
            "sal_qc": 1
        })
    
    anomalies = detector.detect_profile_anomalies(test_measurements)
    
    print(f"Detected {len(anomalies)} anomalies:")
    for a in anomalies:
        print(f"  - {a['type']}: {a['description']} (confidence: {a['confidence']:.2f})")
