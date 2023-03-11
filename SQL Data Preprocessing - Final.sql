--Data from csv file were uploaded on table T_RN177_FNL3 of an Oracle Database

SELECT 

TO_NUMBER(IDXCOL) AS IDXCOL, 
TO_NUMBER(SHIP)AS SHIP,
TO_NUMBER( T) AS T, 
TO_NUMBER(REPLACE(LON,'.',','))LON, 
TO_NUMBER(REPLACE(LAT,'.',','))LAT, 
TO_NUMBER(REPLACE(HEADING,'.',','))HEADING,
TO_NUMBER(REPLACE(COURSE,'.',','))COURSE, 
TO_NUMBER(REPLACE(SPEED,'.',','))SPEED,
STATUS, 
TO_DATE(TDATE,'YYYY-MM-DD HH24:MI:SS') AS TDATE
FROM T_RN177_FNL3
/
SELECT 1.5*3 FROM DUAL;
/

CREATE TABLE T_RN177_FNL4 AS (

SELECT *
/*
T_A AS "var1", LON_A AS "var2", LAT_A AS "var3", HEADING_A AS "var4", COURSE_A AS "var5", SPEED_A AS "var6"
,

TO_NUMBER(REPLACE(LON_A, '.',','))-TO_NUMBER(REPLACE(LON_B, '.',',')) DX, 
TO_NUMBER(REPLACE(LAT_A, '.',','))- TO_NUMBER(REPLACE(LAT_B, '.',',')) DY,
TO_NUMBER(SPEED_A)-TO_NUMBER(SPEED_B) DS*/
FROM(


with alpha as (
SELECT a.*, ROW_number() over(partition by ship order by tdate asc) as rn
from (
SELECT
TO_NUMBER(IDXCOL) AS IDXCOL, 
TO_NUMBER(SHIP)AS SHIP,
TO_NUMBER( T) AS T, 
TO_NUMBER(REPLACE(LON,'.',','))LON, 
TO_NUMBER(REPLACE(LAT,'.',','))LAT, 
TO_NUMBER(REPLACE(HEADING,'.',','))HEADING,
TO_NUMBER(REPLACE(COURSE,'.',','))COURSE, 
TO_NUMBER(REPLACE(SPEED,'.',','))SPEED,
STATUS, 
TO_DATE(TDATE,'YYYY-MM-DD HH24:MI:SS') AS TDATE FROM T_RN177_FNL3
) a
order by ship, tdate
),
beta as (
SELECT
A.RN AS RN_A,
A.IDXCOL AS IDXCOL_A,
A.SHIP AS SHIP_A,
A.T   AS T_A,
A.LON AS LON_A ,
A.LAT AS LAT_A ,
A.HEADING AS HEADING_A, 
A.COURSE  AS COURSE_A, 
A.SPEED   AS SPEED_A , 
A.STATUS   AS STATUS_A , 
A.TDATE   AS TDATE_A,
B.RN AS RN_B,
B.IDXCOL AS IDXCOL_B,
B.SHIP AS SHIP_B,
B.T   AS T_B,
B.LON AS LON_B ,
B.LAT AS LAT_B ,
B.HEADING AS HEADING_B, 
B.COURSE  AS COURSE_B, 
B.SPEED   AS SPEED_B , 
B.STATUS   AS STATUS_B , 
B.TDATE   AS TDATE_B,
round(24*60*(b.tdate-a.tdate),3) as t_interval_minutes
from alpha a, alpha b
where
a.ship=b.ship and 
a.rn=b.rn-1
),

gamma as (
select 
RN_A,IDXCOL_A, SHIP_A, T_A, LON_A , LAT_A ,
 HEADING_A, 
COURSE_A, 
SPEED_A , 
STATUS_A , 
TDATE_A,
RN_B,
IDXCOL_B,
T_B,
LON_B ,
LAT_B ,
HEADING_B, 
COURSE_B, 
SPEED_B , 
STATUS_B , 
TDATE_B,
t_interval_minutes,
AVG(SUM(t_interval_minutes)) OVER (partition by ship_A ORDER BY ship_A, rn_A ) AS moving_average
from beta b 
GROUP BY 
RN_A,IDXCOL_A, SHIP_A, T_A, LON_A , LAT_A ,
 HEADING_A, 
COURSE_A, 
SPEED_A , 
STATUS_A , 
TDATE_A,
RN_B,
IDXCOL_B,
T_B,
LON_B ,
LAT_B ,
HEADING_B, 
COURSE_B, 
SPEED_B , 
STATUS_B , 
TDATE_B,
t_interval_minutes
)
select c.*, 
case when t_interval_minutes > moving_average*10 then 'NEW_ROUTE' ELSE 'NA' END AS MY_FLAG
from gamma c
))
/

---Export Observations for 50k
SELECT 

T_A AS "var1",--Timestamp
REPLACE(TO_CHAR(LON_A), ',','.') AS "var2", 
REPLACE(TO_CHAR(LAT_A), ',','.') AS "var3", 
REPLACE(TO_CHAR(HEADING_A), ',','.') AS "var4", 
REPLACE(TO_CHAR(COURSE_A), ',','.') AS "var5", 
REPLACE(TO_CHAR(SPEED_A), ',','.') AS "var6" /*

TO_NUMBER(REPLACE(LON_A, '.',','))-TO_NUMBER(REPLACE(LON_B, '.',',')) DX, 
TO_NUMBER(REPLACE(LAT_A, '.',','))- TO_NUMBER(REPLACE(LAT_B, '.',',')) DY,
TO_NUMBER(SPEED_A)-TO_NUMBER(SPEED_B) DS
*/
FROM T_RN177_FNL4
WHERE ROWNUM<50017;
/
SELECT TO_CHAR(7, 'fm000')
FROM DUAL;


---Export Actions for 50k
SELECT 
/*
T_A AS "var1",--Timestamp
REPLACE(TO_CHAR(LON_A), ',','.') AS "var2", 
REPLACE(TO_CHAR(LAT_A), ',','.') AS "var3", 
REPLACE(TO_CHAR(HEADING_A), ',','.') AS "var4", 
REPLACE(TO_CHAR(COURSE_A), ',','.') AS "var5", 
REPLACE(TO_CHAR(SPEED_A), ',','.') AS "var6" 
*/
--(LON_A-LON_B),(LAT_A-LAT_B),(SPEED_A-SPEED_B),
REPLACE(TO_CHAR((LON_A-LON_B), '0.999999'), ',','.') DX, 
REPLACE(TO_CHAR((LAT_A-LAT_B), '0.999999'), ',','.') DY,
REPLACE(TO_CHAR((SPEED_A-SPEED_B), '0.999999'), ',','.') DS
FROM T_RN177_FNL4
WHERE ROWNUM<50017;