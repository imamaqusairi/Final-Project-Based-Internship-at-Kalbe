-- NO 1. Berapa Rata - Rata Umur Konsumen Berdasarkan Status Pernikahan
SELECT "Marital Status", AVG(age) as "Average Ages"
FROM publicfinpro."Customer"
GROUP BY "Marital Status";

-- No 2. Berapat Rata-rata umur customer berdasarkan gender 
select gender, avg(age) as "Average Ages"
from publicfinpro."Customer" 
group by gender 

-- No 3. Tentukan Nama Store dengan total quantity terbanyak 
SELECT s.storename, SUM(t.qty) AS "Total Quantity"
FROM publicfinpro."Store" s
INNER JOIN publicfinpro."Transactions" t
ON s.storeid = t.storeid
GROUP BY s.storename
ORDER BY "Total Quantity" DESC
LIMIT 5;

-- No 4. Tentukan Nama Produk Terlari dengan total amount terbanyak
select p."Product Name", SUM(t.totalamount) as "Total Amount"
from publicfinpro."Product" p
inner join publicfinpro."Transactions" t 
on p.productid  = t.productid  
group by p."Product Name"  
order by "Total Amount" desc
limit 5