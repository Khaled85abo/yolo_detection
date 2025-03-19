from typing import Dict, List, Set
import logging

class RuleEngine:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = {
            'overlap': 'ignore',
            'stop': 'ignore',
            'incorrect': 'ignore'
        }
        self.rules_options = ['ignore', 'stop_conveyor', 'alert']
    
    def update_rules(self, rules_data: Dict) -> bool:
        """Update the rules configuration"""
        try:
            self.logger.info(f"Updating rules: {rules_data}")
            
            # Validate the rules data
            if not isinstance(rules_data, dict):
                self.logger.warning("Invalid rules data format")
                return False

            # Update the rules
            for key in self.rules.keys():
                if key in rules_data and rules_data[key] in self.rules_options:
                    self.rules[key] = rules_data[key]
            
            self.logger.info(f"Rules updated: {self.rules}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating rules: {e}")
            return False
    
    def get_rules(self) -> Dict:
        """Get current rules configuration"""
        return {
            'rules': self.rules,
            'rules_options': self.rules_options
        }
    
    def apply_rules(self, detection_types: List[str]) -> Dict:
        """Apply rules for multiple detection types at once"""
        try:
            # Initialize actions
            actions_taken = {
                "stop_conveyor": False,
                "alert": [],
                "ignore": []
            }
            
            # Process each detection type
            for detection_type in detection_types:
                if detection_type not in self.rules:
                    self.logger.warning(f"Unknown detection type: {detection_type}")
                    continue
                    
                rule_action = self.rules[detection_type]
                
                if rule_action == 'stop_conveyor':
                    actions_taken["stop_conveyor"] = True
                    self.logger.info(f"Rule applied: stopping conveyor due to {detection_type}")
                elif rule_action == 'alert':
                    actions_taken["alert"].append(detection_type)
                    self.logger.info(f"Rule applied: alert for {detection_type}")
                elif rule_action == 'ignore':
                    actions_taken["ignore"].append(detection_type)
                    self.logger.info(f"Rule applied: ignoring {detection_type}")
            
            return actions_taken
                
        except Exception as e:
            self.logger.error(f"Error applying rules for detections {detection_types}: {e}")
            return {"error": str(e)}