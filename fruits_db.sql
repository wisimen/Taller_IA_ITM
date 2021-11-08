-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema fruits_db
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema fruits_db
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `fruits_db` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `fruits_db` ;

-- -----------------------------------------------------
-- Table `fruits_db`.`tipo_frutas`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fruits_db`.`tipo_frutas` (
  `idtipo_frutas` INT NOT NULL AUTO_INCREMENT,
  `nombre` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`idtipo_frutas`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `fruits_db`.`frutas`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `fruits_db`.`frutas` (
  `idfrutas` INT NOT NULL AUTO_INCREMENT,
  `ruta` VARCHAR(100) NOT NULL,
  `is_test` BIT(1) NOT NULL,
  `id_tipo_fruta` INT NULL,
  PRIMARY KEY (`idfrutas`),
  INDEX `id_tipo_fruta_fk_idx` (`id_tipo_fruta` ASC) VISIBLE,
  CONSTRAINT `id_tipo_fruta_fk`
    FOREIGN KEY (`id_tipo_fruta`)
    REFERENCES `fruits_db`.`tipo_frutas` (`idtipo_frutas`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
