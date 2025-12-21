"""
Unit tests for aliasing module.
"""

import pytest
import pandas as pd
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.aliasing import (
    FactorMapper,
    GeneratorValidator,
    AliasingEngine,
    parse_generators,
    get_standard_generators,
    format_alias_table,
    validate_resolution_achievable,
    STANDARD_GENERATORS
)


class TestFactorMapper:
    """Test factor name mapping."""
    
    def test_basic_mapping(self):
        """Test basic name to symbol mapping."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS, 
                  ChangeabilityLevel.EASY, levels=[150, 200]),
            Factor("Pressure", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[50, 100]),
            Factor("Time", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[10, 20])
        ]
        
        mapper = FactorMapper(factors)
        
        assert mapper.to_algebraic("Temperature") == "A"
        assert mapper.to_algebraic("Pressure") == "B"
        assert mapper.to_algebraic("Time") == "C"
        
        assert mapper.to_real("A") == "Temperature"
        assert mapper.to_real("B") == "Pressure"
        assert mapper.to_real("C") == "Time"
    
    def test_invalid_factor_name(self):
        """Test that invalid factor name raises error."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[150, 200])
        ]
        
        mapper = FactorMapper(factors)
        
        with pytest.raises(ValueError, match="Unknown factor name"):
            mapper.to_algebraic("Pressure")
    
    def test_invalid_symbol(self):
        """Test that invalid symbol raises error."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[150, 200])
        ]
        
        mapper = FactorMapper(factors)
        
        with pytest.raises(ValueError, match="Unknown symbol"):
            mapper.to_real("Z")
    
    def test_translate_generator(self):
        """Test generator translation."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS, 
                  ChangeabilityLevel.EASY, levels=[150, 200]),
            Factor("Pressure", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[50, 100]),
            Factor("Time", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[10, 20]),
            Factor("Speed", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[5, 15])
        ]
        
        mapper = FactorMapper(factors)
        
        # Translate from real names to algebraic
        result = mapper.translate_generator("Speed=Temperature*Pressure*Time", to_algebraic=True)
        assert result == "D=ABC"
        
        # Translate from algebraic to real names
        result = mapper.translate_generator("D=ABC", to_algebraic=False)
        assert result == "Speed=Temperature*Pressure*Time"
    
    def test_translate_generator_already_algebraic(self):
        """Test that algebraic generators pass through."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS, 
                  ChangeabilityLevel.EASY, levels=[150, 200]),
            Factor("Pressure", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[50, 100]),
            Factor("Time", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[10, 20]),
            Factor("Speed", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[5, 15])
        ]
        
        mapper = FactorMapper(factors)
        
        # If already in algebraic form, recognize it
        # (This is for when user provides generators like "D=ABC")
        result = mapper.translate_generator("Speed=ABC", to_algebraic=True)
        assert result == "D=ABC"


class TestGeneratorValidator:
    """Test generator validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        self.mapper = FactorMapper(self.factors)
        self.validator = GeneratorValidator(5, 1, self.mapper)
    
    def test_valid_generator(self):
        """Test that valid generator passes."""
        # Should not raise
        self.validator.validate_generator_syntax("E=ABCD")
        self.validator.validate_factors_exist("E=ABCD")
    
    def test_missing_equals(self):
        """Test that generator without '=' raises error."""
        with pytest.raises(ValueError, match="must contain '='"):
            self.validator.validate_generator_syntax("EABCD")
    
    def test_empty_left_side(self):
        """Test that empty left side raises error."""
        with pytest.raises(ValueError, match="Left side empty"):
            self.validator.validate_generator_syntax("=ABCD")
    
    def test_empty_right_side(self):
        """Test that empty right side raises error."""
        with pytest.raises(ValueError, match="Right side empty"):
            self.validator.validate_generator_syntax("E=")
    
    def test_multi_character_left_side(self):
        """Test that multi-character left side raises error."""
        with pytest.raises(ValueError, match="must be single factor"):
            self.validator.validate_generator_syntax("EF=ABC")
    
    def test_non_existent_factor(self):
        """Test that referencing non-existent factor raises error."""
        with pytest.raises(ValueError, match="not in base factors"):
            self.validator.validate_factors_exist("E=XYZ")
    
    def test_wrong_number_generators(self):
        """Test that wrong number of generators raises error."""
        with pytest.raises(ValueError, match="Expected 1 generators"):
            self.validator.validate_all(["E=ABCD", "F=ABCE"])


class TestAliasingEngine:
    """Test aliasing computation."""
    
    def test_2_5_1_defining_relation(self):
        """Test defining relation for 2^(5-1) design."""
        engine = AliasingEngine(5, [("E", "ABCD")])
        
        assert "I" in engine.defining_relation
        assert "ABCDE" in engine.defining_relation
    
    def test_2_5_1_resolution(self):
        """Test resolution calculation for 2^(5-1)."""
        engine = AliasingEngine(5, [("E", "ABCD")])
        
        assert engine.resolution == 5
    
    def test_2_4_1_resolution(self):
        """Test resolution calculation for 2^(4-1)."""
        engine = AliasingEngine(4, [("D", "ABC")])
        
        assert engine.resolution == 4
    
    def test_alias_structure_resolution_v(self):
        """Test that Resolution V has clear main effects and 2FI."""
        engine = AliasingEngine(5, [("E", "ABCD")])
        
        # Main effects should be aliased with 4FI or higher
        for letter in "ABCDE":
            if letter in engine.alias_structure:
                aliases = engine.alias_structure[letter]
                for alias in aliases:
                    assert len(alias) >= 4, f"{letter} aliased with {alias}"
    
    def test_word_simplification(self):
        """Test mod-2 word simplification."""
        engine = AliasingEngine(4, [("D", "ABC")])
        
        # A*A = I (should cancel)
        assert engine._simplify_word("AA") == ""
        
        # A*B*A = B (A's cancel)
        assert engine._simplify_word("ABA") == "B"
        
        # ABC*ABC = I (all cancel)
        assert engine._simplify_word("ABCABC") == ""
    
    def test_multiple_generators(self):
        """Test design with multiple generators."""
        engine = AliasingEngine(7, [
            ("E", "ABC"),
            ("F", "BCD"),
            ("G", "ACD")
        ])
        
        # Should have more words in defining relation
        assert len(engine.defining_relation) > 4
        
        # Resolution should be IV
        assert engine.resolution == 4


class TestParseGenerators:
    """Test generator parsing."""
    
    def test_basic_parsing(self):
        """Test basic generator parsing."""
        result = parse_generators(["D=ABC", "E=BCD"])
        
        assert result == [("D", "ABC"), ("E", "BCD")]
    
    def test_with_spaces(self):
        """Test parsing with extra spaces."""
        result = parse_generators(["D = ABC", " E= BCD "])
        
        assert result == [("D", "ABC"), ("E", "BCD")]
    
    def test_invalid_format(self):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError, match="Invalid generator"):
            parse_generators(["DABC"])


class TestStandardGenerators:
    """Test standard generator library."""
    
    def test_common_designs_exist(self):
        """Test that common designs are in library."""
        # 2^(5-1) Resolution V
        assert (5, 1, 5) in STANDARD_GENERATORS
        
        # 2^(7-3) Resolution IV
        assert (7, 3, 4) in STANDARD_GENERATORS
        
        # 2^(8-4) Resolution IV
        assert (8, 4, 4) in STANDARD_GENERATORS
    
    def test_get_standard_generators(self):
        """Test retrieving standard generators."""
        result = get_standard_generators(5, 1, 5)
        
        assert result is not None
        assert result == [("E", "ABCD")]
    
    def test_get_nonexistent_design(self):
        """Test that nonexistent design returns None."""
        result = get_standard_generators(20, 10, 5)
        
        assert result is None


class TestFormatAliasTable:
    """Test alias table formatting."""
    
    def test_basic_formatting(self):
        """Test basic alias table formatting."""
        engine = AliasingEngine(4, [("D", "ABC")])
        
        table = format_alias_table(engine.alias_structure)
        
        # Should be a DataFrame
        assert isinstance(table, pd.DataFrame)
        
        # Should have required columns
        assert 'Effect' in table.columns
        assert 'Aliased_With' in table.columns
        
        # Should have entries
        assert len(table) > 0


class TestValidateResolutionAchievable:
    """Test resolution validation."""
    
    def test_valid_resolution(self):
        """Test that valid resolution passes."""
        generators = [("E", "ABCD")]
        
        # Should not raise
        validate_resolution_achievable(generators, 5, 5)
    
    def test_invalid_resolution(self):
        """Test that claiming higher resolution than achievable raises error."""
        generators = [("D", "ABC")]  # Resolution IV
        
        with pytest.raises(ValueError, match="achieve Resolution 4, not 5"):
            validate_resolution_achievable(generators, 5, 4)


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete workflow from factors to alias structure."""
        # Create factors with real names
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[150, 200]),
            Factor("Pressure", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[50, 100]),
            Factor("Time", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[10, 20]),
            Factor("Speed", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[100, 200]),
            Factor("Catalyst", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[1, 5])
        ]
        
        # Create mapper
        mapper = FactorMapper(factors)
        
        # Validate generators
        validator = GeneratorValidator(5, 1, mapper)
        validator.validate_all(["E=ABCD"])
        
        # Parse generators
        generators = parse_generators(["E=ABCD"])
        
        # Build aliasing structure
        engine = AliasingEngine(5, generators)
        
        # Verify results
        assert engine.resolution == 5
        assert "ABCDE" in engine.defining_relation
        
        # Format table
        table = format_alias_table(engine.alias_structure)
        assert isinstance(table, pd.DataFrame)